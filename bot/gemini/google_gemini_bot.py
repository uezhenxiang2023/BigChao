"""
Google gemini bot

@author guofu
@Date 2026/03/27
"""
# encoding:utf-8

import base64
import os, time
import shutil
import tempfile

from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, Part, FunctionDeclaration, Type, FileData

from config import conf
from bot.bot import Bot
from bot.gemini.gemini_common import (
    data_url_to_part,
    get_image_context_from_session,
    mark_image_context_injected,
    should_inject_image_context,
)
from bot.gemini.gemini_error import format_gemini_error, is_gemini_sdk_error
from bot.gemini.google_gemini_session import _gemini_sessions

from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from common import const, memory
from common.tool_button import tool_state
from common.model_status import model_state

from plugins.bigchao.script_breakdown import screenplay_scenes_breakdown, screenplay_assets_breakdown, screenplay_formatter


# OpenAI对话模型API (可用)
class GoogleGeminiBot(Bot):
    _GEMINI_INLINE_PDF_LIMIT_BYTES = 20 * 1024 * 1024
    _GEMINI_MAX_PDF_LIMIT_BYTES = 50 * 1024 * 1024

    def __init__(self):
        super().__init__()
        self.api_key = conf().get("gemini_api_key")
        self.api_key_paid = conf().get("gemini_api_key_paid")
        """self.model = conf().get('model')
        self.Model_ID = self.model.upper()
        self.image_model = conf().get('text_to_image')
        self.IMAGE_MODEL_ID = self.image_model.upper()"""
        self.system_prompt = conf().get("character_desc")
        self.stream = conf().get("stream")
        self.function_call_dicts = {
            "screenplay_scenes_breakdown": screenplay_scenes_breakdown,
            "screenplay_assets_breakdown": screenplay_assets_breakdown,
            'screenplay_formatter': screenplay_formatter
        }
        # 调用gemini自己的token计算方式
        self.sessions = _gemini_sessions
        self.generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        self.tool_config={'function_calling_config': 'AUTO'}

        # Default client; paid models will switch to the paid key dynamically.
        self.client = genai.Client(api_key=self.api_key)
         # schema for screenplay_scenes_breakdown need to be updated
        self.screenplay_scenes_breakdown_schema = FunctionDeclaration(
            name="screenplay_scenes_breakdown",
            description="拆解剧本一共两个步骤，这是第一步，拆解场景。阅读剧本，结合内容，先提取剧本名称和剧本总页数，再逐个拆解场景，提取场号、场景名称、内外、日夜等基础信息，做成场景列表",
            parameters=types.Schema(
                type=Type.OBJECT,
                required=["screenplay_title", "scenes_list"],
                properties={
                    "screenplay_title": types.Schema(
                        type=Type.STRING,
                        description="从文档中提取出的剧本名称",
                    ),
                    "total_pages": types.Schema(
                        type=Type.INTEGER,
                        description="剧本总页数",
                    ),
                    "total_words": types.Schema(
                        type=Type.INTEGER,
                        description="剧本总字数",
                    ),
                    "scenes_list": types.Schema(
                        type=Type.ARRAY,
                        items=types.Schema(
                            type=Type.OBJECT,
                            properties={
                                "scene_id": types.Schema(
                                    type=Type.INTEGER,
                                    description="场号，包含正戏与彩蛋，彩蛋场号顺接正戏，比如正戏最后一场是95，彩蛋第一场就是96",
                                ),
                                "location": types.Schema(
                                    type=Type.STRING,
                                    description="场景名称，包含正戏与彩蛋",
                                ),
                                "daynight": types.Schema(
                                    type=Type.STRING,
                                    description="日景还是夜景，如果剧本中某一场次开头没有明确标注，根据该场次内容自行推断，并从枚举列表中提取相应推断结果",
                                    enum = ["日", "夜", "极昼", "极夜", "晨", "昏", "晨(极昼)", "夜(极昼)", "晨(极夜)", "夜(极夜)"],
                                ),
                                "envirement": types.Schema(
                                    type=Type.STRING,
                                    description="室内环境还是室外环境，如果剧本中某一场次开头没有明确标注，根据该场次内容自行推断，并从枚举列表中提取相应推断结果",
                                    enum = ["内", "外", "外/内", "内/外"],
                                ),
                                "words": types.Schema(
                                    type=Type.INTEGER,
                                    description="场次字数",
                                ),
                                "pages": types.Schema(
                                    type=Type.NUMBER,
                                    description="场次页数，计算方式为场次字数除以页均字数即(words / (total_words / total_pages))",
                                ),
                                "estimated_duration": types.Schema(
                                    type=Type.NUMBER,
                                    description="场次预计时长，单位是分钟，计算方式为场次页数的1.2倍即(pages * 1.2)",
                                ),
                                "assets_id": types.Schema(
                                    type=Type.ARRAY,
                                    description="场次引用资产的id，从screenplay_assets_breakdown的返回值中取",
                                    items=types.Schema(
                                        type = Type.STRING
                                    )
                                ),
                            },
                        ),
                    ),
                },
            ),
        )
        self.screenplay_assets_breakdown_schema = FunctionDeclaration(
            name="screenplay_assets_breakdown",
            description="拆解剧本一共分两个步骤，这是第二步，拆解资产。阅读剧本，熟悉内容，逐场提取场景、人物和道具名称，制作成资产列表。该工具需要跟sreenplay_scenes_breakdown同时平行调用，才能完成剧本拆解。",
            parameters=types.Schema(
                type=Type.OBJECT,
                required=["screenplay_title", "assets_list"],
                properties={
                    "screenplay_title": types.Schema(
                        type=Type.STRING,
                        description="剧本名称，调用函数必需输入的参数，仔细阅读文档，提取出剧本名称",
                    ),
                    "assets_list": types.Schema(
                        type=Type.ARRAY,
                        items=types.Schema(
                            type=Type.OBJECT,
                            required=['asset_id'],
                            properties={
                                "asset_type": types.Schema(
                                    type = types.Type.STRING,
                                    description = "场次包含的资产类型，由人、物和事(过程)三大类构成。cast属于人，是有名字、有台词、在很多场次中重复出现的角色，没有台词但对剧情发展起到重要作用的人物也属于cast；stunts属于事(过程)，通常是指危险系数较高的动作戏，比如在身上着火、从楼梯上滚下来、爬上4米高的云梯、在索道上滑行、从自行车上摔落等，由特技或动作演员完成，搬家、看房、卖车等普通动作构成的事件不包含在stunts中；extra_silent属于人，是没名字、没台词、不影响剧情发展的人物，这些人物在剧本中通常以身份、职业或功能的形式出现，如1个调酒师，1位女服务员，部门领导等；extra_atmosphere属于人，是出现在背景或环境中的群众演员，比如10个商人、8个排队的游客等；costume是服装，属于物，包括鞋袜和帽子以及服饰的局部描述，如白色衬衣、牛仔裤、外套内兜、运动鞋、渔夫帽等；makeup是伤口、血、假肢等特殊化妆，属于事(过程)；location属于物，是不需要陈设或仅需简单陈设的外景；set_dressing属于事(过程)，是需要精心设计与陈设的内景；vehicle属于物，是戏用交通工具；animal属于物，是需要驯养员在拍摄现场看守与训练的宠物或者小型动物；livestock属于物，是需要交通工具来运输的大型动物，比如牛、马等;prop属于物，戏用道具;special_effects属于事(过程)，是需要在现场完成的烟火、爆破、撞车等特效，镜头块切这类剪辑效果或慢镜头这类摄影技巧不包含在special_effects中",
                                    enum = [
                                    "cast",
                                    "stunts",
                                    "extra_silent",
                                    "extra_atmosphere",
                                    "costume",
                                    "makeup",
                                    "location",
                                    "set_dressing",
                                    "vehicle",
                                    "animal",
                                    "livestock",
                                    "prop",
                                    "special_effects"
                                    ],
                                ),
                                "visual_effects_type": types.Schema(
                                    type = types.Type.STRING,
                                    description = "资产所对应的镜头是否需要后期视效(VFX)，digital_envrionment对应location或set_dressing,是数字环境，剧本中通常用魔幻、奇幻、超现实这样的关键词来描这类场景；digital_cast对应cast，是有姓名、性格、台词和故事线的数字角色；digital_double对应cast或stunts,是真人演员的数字替身，坠崖、车祸等对演员生命安全有威胁的戏份需要后期视效用演员的数字替身来完成；digital_creature对应animal或livestock,是数字生物，通常是现实世界中不存在的动物；set_extension对应location或set_dressing，是场景的数字延伸，如窗户外的特殊环境或街道环境的延伸；screen_comp对应prop，是电视机、电脑或手机屏幕内容的数字合成；sky_replacement对应location,是霞光、极光、极昼、极夜、日落、日出等特殊条件下的天空替换；digital_weather对应location,是台风、暴雨、暴雪、闪电等数字气象效果；digital_makeup对应makeup,是伤痕、血迹、假肢、残肢等数字特效化妆；digital_crowds对应extra_atmosphere或livestock,是数字集群，如体育场看台上数以万计的观众、草原上几千匹奔腾的骏马、铺天盖地的蝴蝶群等；day_for_night对应location,是通过后期视效将日景转为夜景；digital_prop对应prop,是数字道具，in_vehicle_comp对应vehicle，是交通工具窗户外移动背景合成，通常是拍摄演员在静止不动的交通工具内部，如开车、开飞机或开船；simulation_fx对应stunts,是数字弹道、爆破、火焰、浓烟、坍塌、破碎等效果",
                                    enum = [
                                    "digital_envrionment",
                                    "digital_cast",
                                    "digital_double",
                                    "digital_creature",
                                    "set_extension",
                                    "screen_comp",
                                    "sky_replacement",
                                    "digital_weather",
                                    "digital_makeup",
                                    "digital_crowds",
                                    "day_for_night",
                                    "digital_prop",
                                    "in_vehicle_comp",
                                    "simulation_fx"
                                    ],
                                ),
                                "visual_effects_description": types.Schema(
                                    type=Type.STRING,
                                    description="visua_effects内容阐述",
                                ),
                                "asset_id": types.Schema(
                                    type=Type.STRING,
                                    description="资产ID，格式为资产类型+两位阿拉伯数字即asset_type##，如cast01，location02，prop03, vehicle04等",
                                ),
                                "name": types.Schema(
                                    type=Type.STRING,
                                    description="资产名称,完整保留每一场中出现的资产，包括无名无姓的角色、一闪而过的场景、无足轻重的道具，不要简化或过滤。",
                                ),
                                "scene_ids": types.Schema(
                                    type=Type.ARRAY,
                                    description="资产所出现的场次",
                                    items=types.Schema(
                                        type = Type.INTEGER
                                    )
                                ),
                                "asset_pages": types.Schema(
                                    type=Type.NUMBER,
                                    description="资产所出现场次总页数",
                                ),
                                "estimated_asset_duration": types.Schema(
                                    type=Type.NUMBER,
                                    description="预计资产所出现场次总时长，单位是分钟，计算方式为资产页数的1.2倍即(asset_pages * 1.2)",
                                )
                            },
                        ),
                    ),
                },
            ),
        )
        self.screenplay_formatter = types.FunctionDeclaration(
            name="screenplay_formatter",
            description="按照好莱坞电影工业的标准，对剧本进行排版。",
            parameters=types.Schema(
                type = Type.OBJECT,
                required = ["screenplay_title", "screenwriter", "paragraphs_metadata"],
                properties = {
                    "screenplay_title": types.Schema(
                        type = Type.STRING,
                        description = "从文档中提取出的剧本名称，如果没有，需要请用户提供",
                    ),
                    "screenwriter": types.Schema(
                        type = Type.STRING,
                        description = "从文档中提取出的剧本名称,如果没有，需要请用户提供",
                    ),
                    "paragraphs_metadata": types.Schema(
                        type = Type.ARRAY,
                        description = "剧本段落的原始数据",
                        items = types.Schema(
                            type = Type.OBJECT,
                            properties = {
                                "scene_heading": types.Schema(
                                    type = Type.OBJECT,
                                    description = "场次标题的结构化对象，将场号、环境、场景名称和时间从段落中提取出来存储在结构化的对象中，方便后续调用API时传递参数。该值不能为空值",
                                    properties = {
                                        "scene_id": types.Schema(
                                            type = Type.INTEGER,
                                            description = "段落所在的场号，包含正戏与彩蛋，彩蛋场号顺接正戏，比如正戏最后一场是95，彩蛋第一场就是96",
                                        ),
                                        "enviroment": types.Schema(
                                            type = Type.STRING,
                                            description = "段落发生的环境，大部分情况是内或外，内/外或外/内表明该场次同时包含了室内外的景，比如说车戏的时候，或者是门、窗内外的角色有互动的时候。如果场次标题中没有环境信息，根据段落内容进行推测。",
                                            enum = ["内", "外", "内/外", "外/内"],
                                        ),
                                        "location": types.Schema(
                                            type = Type.STRING,
                                            description = "段落发生的场景名称，包含正戏与彩蛋",
                                        ),
                                        "daynight": types.Schema(
                                            type = Type.STRING,
                                            description = "段落发生的时间，如果场次标题中没有环境信息，根据段落内容进行推测。",
                                            enum = ["日", "夜"],
                                        ),
                                    },
                                ),
                                "paragraph_id": types.Schema(
                                    type = Type.INTEGER,
                                    description = "段落序号，段落通常以句号'。'、感叹号‘！’、问号‘？’、冒号‘：’或段落标记号‘¶’结束。",
                                ),
                                "content": types.Schema(
                                    type = Type.STRING,
                                    description = "从文档中提取出的段落内容",
                                ),
                                "category": types.Schema(
                                    type = Type.STRING,
                                    description = "段落类别，scene_setting是场景段落，通常出现在每一场的第一行，内容描写顺序是内外环境，场景名称，时间；action是动作段落，场景和事物的客观描述，通常是以句号结束；character是角色段落（特别注意，旁白、画外音也属于角色段落），通常以姓名或身份加冒号‘：’结束，比如“孙涛：”或“产房护士：”；有时会在角色名字或身份名和冒号‘：’之间加入动作或情绪描述，如“张智宇笑到：”，此时冒号前部分也是角色段落；有时character段落以“旁白或画外音”加冒号‘：’结束，如“旁白:”或；“央媒新闻播报的画外音”；偶尔会用字母代替角色或身份加冒号‘：’，如“A：你怎么来了”、“C:我来看看你和小丽”；有的情况需要在角色名、身份名或‘画外音’后加Extension扩展，比如（V.O.）和（O.S.），VO即Voice Over，场景之外角色的声音——旁白或独白，这个好理解，OS即Off Screen是指角色不在镜头内发出的声音，比如说一个角色在书房写作业，这时他的母亲在厨房大喊一声“出来吃饭了”，这就是OS；偶尔剧本内容提及屏幕出现的字幕内容，也会用冒号‘：’作为结束，如“黑屏，出字幕:根据国家相关法律法规，非发集资，教唆别人炒房，从中放高利贷，非法吸收存款，证券化炒房，属于扰乱金融市场，均构成金融犯罪。”这种情况，冒号前是action段落，冒号后是dialogue段落；dialogue是台词段落，通常紧跟角色段落；特别注意，画外音和屏幕字幕提示也属于dialogue段落",
                                    enum = ["scene_setting", "action", "character", "dialogue"],
                                ),
                            },
                        )
                    ),
                },
            ),
        )
        self.function_declarations = Tool(
            function_declarations=[
                self.screenplay_scenes_breakdown_schema,
                self.screenplay_assets_breakdown_schema,
                self.screenplay_formatter
                ]
        )
        self.google_search_tool = Tool(google_search=GoogleSearch())

        self.search_config = GenerateContentConfig(
            system_instruction=self.system_prompt,
            safety_settings=self.safety_settings,
            tools=[self.google_search_tool],
            response_modalities=['TEXT'],
            **self.generation_config
        )

        self.print_config = GenerateContentConfig(
            system_instruction=self.system_prompt,
            safety_settings=self.safety_settings,
            tools=[self.function_declarations],
            tool_config={
                'function_calling_config': {
                    'mode': 'ANY',
                    "allowedFunctionNames": ['screenplay_formatter']
                }
            },
            response_modalities=['TEXT'],
            **self.generation_config
        )

        self.breakdown_config = GenerateContentConfig(
            system_instruction=self.system_prompt,
            safety_settings=self.safety_settings,
            tools=[self.function_declarations],
            tool_config={
                'function_calling_config': {
                    'mode': 'ANY',
                    "allowedFunctionNames": ["screenplay_scenes_breakdown", "screenplay_assets_breakdown"]
                }
            },
            response_modalities=['TEXT'],
            **self.generation_config
        )
        # 用字典存储用户会话实例
        self.user_chats = {}
        # Chat keeps using the client that created it.  Retain that client for
        # the lifetime of the cached chat; replacing the only client reference
        # can close its underlying HTTP transport between messages.
        self.user_chat_clients = {}

    def _get_client_by_model(self, model):
        api_key = self.api_key_paid if model in const.GEMINI_PAID else self.api_key
        return genai.Client(api_key=api_key)

    def _get_user_chat(self, session_id, model):
        """获取指定用户的chat实例,如果不存在则创建新的"""
        chat_key = f"{session_id}:{model}"
        if chat_key not in self.user_chats:
            client = self._get_client_by_model(model)
            self.user_chat_clients[chat_key] = client
            self.user_chats[chat_key] = client.chats.create(
                model=model,
                config=GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    safety_settings=self.safety_settings,
                    tools=[self.function_declarations],
                    tool_config={
                        'function_calling_config': {
                            'mode': 'NONE'
                        }
                    },
                    response_modalities=['TEXT'],
                    **self.generation_config
                )
            )
        self.client = self.user_chat_clients[chat_key]
        return self.user_chats[chat_key]

    def _build_request_contents(self, query: str, session_id: str):
        """
        构建请求内容列表，包括处理文本和缓存的媒体文件
        
        :param query: 用户查询文本
        :param session_id: 用户会话ID
        :return: 构建好的请求内容列表
        """
        # 如果是打印状态，修改查询内容
        if tool_state.get_print_state(session_id):
            query = f"大超子，帮忙给这个本子排版，编剧的名字是{query}"
        
        # 创建文本Part对象并添加到列表
        text = Part.from_text(text=query)
        request_contents = [text]
        request_warnings = []
        
        image_cache = memory.USER_IMAGE_CACHE.get(session_id)
        if image_cache:
            first_data = image_cache['files'][0]
            data_type = type(first_data).__name__

            if data_type in ['JpegImageFile', 'PngImageFile', 'File']:
                request_contents.extend(image_cache['files'])
            elif data_type in ['FileData']:
                request_contents.append({'fileData': first_data})

            memory.USER_IMAGE_CACHE.pop(session_id)

        file_cache = memory.USER_FILE_CACHE.get(session_id)
        if file_cache:
            for cached_file in file_cache.get("files", []):
                if not isinstance(cached_file, dict):
                    logger.warning(f"[{self.Model_ID}] unsupported cached file type: {type(cached_file).__name__}")
                    continue

                mime_type = cached_file.get("mime_type")
                raw_data = cached_file.get("data", "")
                file_path = cached_file.get("path", "")

                if mime_type in ['application/docx', 'application/doc', 'application/plain']:
                    request_contents.insert(0, Part.from_text(text=raw_data))
                    logger.debug(f"[{self.Model_ID}] document text part added, mime_type={mime_type}")
                elif mime_type == 'application/pdf':
                    file_content = self._build_pdf_part(file_path, raw_data, request_warnings)
                    if file_content is not None:
                        request_contents.insert(0, file_content)
                else:
                    logger.warning(f"[{self.Model_ID}] unsupported file mime_type: {mime_type}")

            memory.USER_FILE_CACHE.pop(session_id)

        if not image_cache and not file_cache:
            image_context = get_image_context_from_session(session_id)
            session_images = image_context["images"]
            if session_images and should_inject_image_context(session_id, image_context["signature"]):
                request_contents.extend([data_url_to_part(image_url) for image_url in session_images])
                if image_context["prompt"]:
                    request_contents.append(
                        Part.from_text(
                            text=f"补充背景信息（仅供参考，不要复述给用户）：这张图此前可能由这个提示词生成：{image_context['prompt']}"
                        )
                    )
                mark_image_context_injected(session_id, image_context["signature"])
                logger.info(f"[{self.Model_ID}] 从 session 历史注入图片上下文, count={len(session_images)}, has_prompt={bool(image_context['prompt'])}")
        
        return request_contents, request_warnings

    def _build_pdf_part(self, file_path: str, raw_data: str, request_warnings=None):
        file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
        if file_size > self._GEMINI_MAX_PDF_LIMIT_BYTES:
            warning = "Gemini 官方目前只支持 50MB 以内的 PDF，请压缩或拆分后再试。"
            logger.warning(
                f"[{self.Model_ID}] skip oversized PDF, path={file_path}, size={file_size}, "
                f"limit={self._GEMINI_MAX_PDF_LIMIT_BYTES}"
            )
            if request_warnings is not None and warning not in request_warnings:
                request_warnings.append(warning)
            return None
        if file_size > self._GEMINI_INLINE_PDF_LIMIT_BYTES and file_path:
            logger.info(f"[{self.Model_ID}] PDF exceeds inline threshold, upload to Gemini Files API, path={file_path}")
            return self.upload_to_gemini(file_path, mime_type='application/pdf')

        logger.debug(f"[{self.Model_ID}] PDF document part added")
        return Part.from_bytes(
            data=base64.b64decode(raw_data),
            mime_type='application/pdf'
        )

    def reply(self, query, context: Context = None) -> Reply:
        try:
            session_id = context["session_id"]
            self.model = model_state.get_basic_state(session_id)
            self.Model_ID = self.model.upper()

            if context.type == ContextType.TEXT:
                logger.info(f"[{self.Model_ID}] query={query}, requester={session_id}")
                if self.model not in const.GEMINI_GENAI_SDK:
                    logger.warning(f"[{self.Model_ID}] Unsupported Gemini model")
                    return Reply(ReplyType.ERROR, f"[{self.Model_ID}] Unsupported Gemini model")

                session = self.sessions.session_query(query, session_id)
                user_chat = self._get_user_chat(session_id, self.model)
                resquest_contents, request_warnings = self._build_request_contents(query, session_id)
                if len(resquest_contents) == 1 and request_warnings:
                    raise ValueError("\n".join(request_warnings))
                if tool_state.get_print_state(session_id):
                    response = user_chat.send_message(resquest_contents, config=self.print_config)
                    tool_state.toggle_printing(session_id)
                elif tool_state.get_breakdown_state(session_id):
                    response = user_chat.send_message(resquest_contents, config=self.breakdown_config)
                    tool_state.toggle_breakdowning(session_id)
                else:
                    is_searching = tool_state.get_search_state(session_id)

                    if self.stream:
                        logger.info(f"[{self.Model_ID}] stream 模式已开启")

                        def stream_generator():
                            full_text = ""
                            final_response = None
                            try:
                                if is_searching:
                                    stream_response = user_chat.send_message_stream(
                                        resquest_contents, config=self.search_config
                                    )
                                else:
                                    stream_response = user_chat.send_message_stream(resquest_contents)

                                for chunk in stream_response:
                                    if chunk.text:
                                        full_text += chunk.text
                                        yield chunk.text
                                    final_response = chunk

                                self.sessions.session_reply(full_text, session_id)
                                logger.info(f"[{self.Model_ID}] stream 完成, session_id={session_id}")

                                if is_searching and final_response is not None:
                                    grounding_metadata = getattr(
                                        final_response.candidates[0], "grounding_metadata", None
                                    ) if final_response.candidates else None
                                    if grounding_metadata and grounding_metadata.grounding_chunks:
                                        yield final_response
                            except Exception as e:
                                logger.error(f"[{self.Model_ID}] stream fetch reply error, {e}")
                                if is_gemini_sdk_error(e):
                                    error_text = format_gemini_error(e, self.model)
                                else:
                                    error_text = f"[{self.Model_ID}] {e}"
                                full_text += error_text
                                yield error_text

                        return Reply(ReplyType.STREAM, stream_generator())
                    else:
                        response = user_chat.send_message(resquest_contents)
                if response.text == None and response.function_calls == []:
                    logger.warning(f"[{self.Model_ID}] Response Text is None")
                    return Reply(ReplyType.ERROR, f"[{self.Model_ID}] Response Text is None")
                response, function_response = self.function_call_polling_loop(session_id, response, user_chat)

                # 是否开启联网搜索
                if not tool_state.get_search_state(session_id):
                    # 是否监测到函数响应内容
                    if function_response != []:
                        response = {
                            'reply_text': ("\n".join(request_warnings) + "\n\n" if request_warnings else "") + response.text,
                            'function_response': function_response[0].function_response.response
                        }
                        return Reply(ReplyType.FILE, response)
                    else:
                        reply_text = ("\n".join(request_warnings) + "\n\n" if request_warnings else "") + response.text
                        return Reply(ReplyType.TEXT, reply_text)
                else:
                    grounding_metadata = response.candidates[0].grounding_metadata
                    # 响应中是否有网页链接
                    if grounding_metadata.grounding_chunks is None:
                        reply_text = ("\n".join(request_warnings) + "\n\n" if request_warnings else "") + response.text
                        return Reply(ReplyType.TEXT, reply_text)
                    else:
                        return Reply(ReplyType.IMAGE_URL, response)
            else:
                logger.warning(f"[{self.Model_ID}] Unsupported message type, type={context.type}")
                return Reply(ReplyType.ERROR, f"[{self.Model_ID}] Unsupported message type, type={context.type}")
        except Exception as e:
            logger.error("[{}] fetch reply error, {}".format(self.Model_ID, e))
            if is_gemini_sdk_error(e):
                return Reply(ReplyType.ERROR, format_gemini_error(e, self.model))
            return Reply(ReplyType.ERROR, f"[{self.Model_ID}] {e}")

    def function_call_reply(self, part):
        if hasattr(part, 'function_call'):
            fn = part.function_call
            fn_dict = type(fn).to_dict(fn)
            fn_name = fn_dict.get('name')
            fn_args = fn_dict.get('args')
        else:
            fn_name = part.name
            if "assets" in fn_name.lower():
                fn_name = "screenplay_assets_breakdown"
            fn_args = part.args
        function_call_reply = {
            "functionCall": {
                "name": fn_name,
                "args": fn_args
            }
        }
        return function_call_reply

    def function_call_polling_loop(self, session_id, response, user_chat):
        """轮询模型响应结果中的函数调用"""
        function_calling = True
        function_response = []
        while function_calling:
            if hasattr(response, 'function_calls'):
                function_calls = response.function_calls if response.function_calls is not None else []
            else:
                # 为gemini 2.0以前的版本留出调用接口
                function_calls = response.parts if response.parts[0].function_call.args is not None else []
            function_response_parts = []
            for part in function_calls:
                function_call_reply = self.function_call_reply(part)
                fn_name = function_call_reply.get('functionCall').get('name')
                fn_args = function_call_reply.get('functionCall').get('args')
                """# 将reply_text转换为字符串 --- gemini 2.0开始，chat方法自动维护消息历史，后续考虑暂停维护脚手架中的旧版gemini的消息历史
                function_call_str = json.dumps(function_call_reply)
                # add function call to session as model/assistant message
                self.sessions.session_reply(function_call_str, session_id)"""
                # call function
                function_call = self.function_call_dicts.get(fn_name)
                # 从fn_args中获取function_call的参数
                function_response_part = function_call(session_id, fn_name, **fn_args)
                function_response_parts.extend(function_response_part)
                function_response = function_response_part
                """# add function response to session as user message --- gemini 2.0开始，chat方法自动维护消息历史，后续考虑暂停维护脚手架中旧版gemini的消息历史
                self.sessions.session_query(function_response_str, session_id)"""
            if function_calls:
                # new turn of model request with function response
                response = user_chat.send_message(function_response_parts)
                continue
            function_calling = False
        return response, function_response

    def _file_cache(self, context):
        memory.USER_FILE_CACHE[context['session_id']] = {
            "path": context.content,
            "msg": context.get("msg")
        }
        logger.info("[{}] file={} is cached for assistant".format(self.Model_ID, context.content))
        return None

    def _file_download(self, file_cache):
        msg = file_cache.get("msg")
        path = file_cache.get("path")
        msg.prepare()
        logger.info("[{}] file={} is downloaded locally".format(self.Model_ID, path))
        return None

    def cache_media(self, media_path, media_file, context):
        session_id = context["session_id"]
        if session_id not in memory.USER_IMAGE_CACHE:
            memory.USER_IMAGE_CACHE[session_id] = {
                "path": [media_path],
                "files": [media_file]
            }
        else:
            memory.USER_IMAGE_CACHE[session_id]["path"].append(media_path)
            memory.USER_IMAGE_CACHE[session_id]["files"].append(media_file)
        logger.info(f"[{self.model}] {media_path} cached to memory")
        return None

    def upload_to_gemini(self, path, mime_type=None):
        """Uploads the given file to Gemini.

        https://ai.google.dev/gemini-api/docs/prompting_with_media
        """
        upload_config = {}
        if mime_type:
            upload_config["mime_type"] = mime_type
        temp_path = None
        upload_path = path
        try:
            os.path.basename(path).encode("ascii")
        except UnicodeEncodeError:
            suffix = os.path.splitext(path)[1] or ".bin"
            temp_dir = os.path.join(os.getcwd(), "tmp", "gemini_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix="gemini_upload_", suffix=suffix, dir=temp_dir, delete=False) as tmp_file:
                temp_path = tmp_file.name
            shutil.copyfile(path, temp_path)
            upload_path = temp_path
            logger.info(f"[{self.Model_ID}] normalized upload filename for Gemini Files API, tmp_path={upload_path}")
        file = self.client.files.upload(
            file=upload_path,
            config=upload_config or None
        )
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        self.wait_for_files_active(file)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def wait_for_files_active(self, media_file):
        """Waits for the given files to be active.

        Some files uploaded to the Gemini API need to be processed before they can be
        used as prompt inputs. The status can be seen by querying the file's "state"
        field.

        This implementation uses a simple blocking polling loop. Production code
        should probably employ a more sophisticated approach.
        """
        print("Waiting for file processing...")
        file_name =media_file.name
        file = self.client.files.get(name=file_name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = self.client.files.get(name=file_name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
        print("...all files ready")
        print()
