"""Build SYNTHETIC extra cases (v2) for the RWKV-state decoder pilot.

WHAT THIS IS
------------
Every case this script writes is **explicitly synthetic**: template-generated
Chinese event/assistant turns plus hand-designed slot templates, over a table of
50 invented engineering domains. No real event, log, memory file or dialogue was
copied or quoted. The generated topics are *not* real work the user did; they
exist so the four-slot decoder sees varied wording, varied negations and varied
constraint patterns without burning real heldout material.

It emits a JSONL file that the pilot reads through `rwkv_slots_data.load_events`
(`train_rwkv_slots.py --cases <file>`). It writes only its own two artifacts:
the cases file and a metadata/report file beside it.

SIX CASE KINDS PER DOMAIN (labels stay honest, checked by DATA.validate)
  pending      unverified, 未完成 filled          -> 结果 must hedge
  failed       unverified, an error is unresolved -> 结果 must hedge
  completed    verified, confirmed measurement    -> 结果 must contain 已确认
  correction   verified, an earlier value is retracted
  constraint   verified, a standing policy is pinned
  unlanded     unverified, a rule recorded but not wired in yet

DATA GATES THE GENERATOR MUST SURVIVE (rwkv_slots_data.validate)
  * exactly the four slots in the fixed order, all non-empty
  * summary <= 200 chars, topic <= 14 chars
  * unique case ids, unique summary text
  * unverified case: 结果 must contain an explicit hedge
  * verified case: 结果 must contain 已确认/已完成, and may not assert as done an
    action the source marks undone
  * a non-topic slot value of >= 8 chars must NOT be a verbatim substring of the
    source turn (the decoder must not be able to copy its answer)
The extra checks below go further: a longest-common-substring probe reports how
much wording any assertion slot shares with its source and attributes the shared
runs to the (deliberately copyable) topic noun phrase.

SYNTHETIC LIMITS -- do not quote results as if they came from real events
  * Template-generated: wording comes from a small pool of phrasings rotated by
    index, so surface variety is bounded. A decoder can learn these templates
    instead of reading the recurrent state; that is exactly the failure mode the
    shuffled/no-state ablations are there to expose.
  * Train and heldout extras are drawn from the SAME kind templates and merely
    use disjoint domains (40 train domains, 10 heldout domains). Template
    overlap therefore survives across the split -- synthetic heldout is weaker
    evidence than real heldout, and must never be reported as if equivalent.
  * Multi-turn cases encode earlier turns as TEXT INSIDE the user field
    (`【历史轮次】/【当前轮次】`). The frozen backbone reads one concatenated
    string, so this does NOT exercise multi-turn recurrence; it is pilot-level
    source variety only.
  * Per-case `synthetic`/`kind`/`generator` fields are file-level provenance:
    `rwkv_slots_data._normalized` keeps only id/group/user/assistant/unverified/
    summary, so the trainer never sees them. Regenerate with this script when
    auditing; do not assume the loader carries the flag through.
  * ids are prefixed S (extra train) / V (extra heldout) and can never collide
    with the embedded T01-T16 / H01-H06.

Run:
  py -3.14 build_rwkv_training_v2.py                 # write + validate + report
  py -3.14 build_rwkv_training_v2.py --check-only    # re-validate an existing file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rwkv_slots_data as DATA

SCRIPT_VERSION = "build_rwkv_training_v2.py/1"
GENERATOR = "build_rwkv_training_v2.py"
LABEL = "SYNTHETIC template-generated extras for the decoder-only pilot"
SYNTHETIC = True

DEFAULT_OUT = Path(r"D:\AI\DBA\training\rwkv-slots-v2\cases.jsonl")
DEFAULT_META = Path(r"D:\AI\DBA\training\rwkv-slots-v2\cases.meta.json")
# A template.json (or DBA_TRAIN_TEMPLATE) silently takes precedence over --cases
# inside load_events, so the checks below force the JSONL branch explicitly.
NO_TEMPLATE = DEFAULT_OUT.with_name("__no_template__.json")

TRAIN_DOMAINS = 40          # 40 domains x 6 kinds = 240 extra train cases
HELDOUT_DOMAINS = 10        # 10 domains x 6 kinds =  60 extra heldout cases
LCS_LIMIT = 8               # the length DATA.validate already treats as copying

# Topics/vocabulary that must not appear anywhere in the generated text: the
# six heldout subjects (no flying firmware, battery, camera, translation,
# masking, driver) plus the embedded train subjects (indices, training runs,
# retrieval paths, cron, line endings, memory migration, vectors, inference
# speed). Keeping both clear means v2 cannot silently overlap an existing topic.
FORBIDDEN_TERMS = (
    "飞控", "固件", "电池", "相机", "标定", "翻译", "脱敏", "掩码", "masking",
    "驱动", "显卡", "索引", "训练", "推理", "检索", "记忆", "定时任务", "行尾",
    "向量", "断电", "续训",
)
HELDOUT_TOPICS = tuple(c["summary"].split("\n")[0].split("：", 1)[1]
                       for c in DATA.HELDOUT_CASES)


class Domain:
    """One invented engineering domain and its per-kind vocabulary."""

    __slots__ = ("key", "keep", "protect", "act", "metric", "val", "val2", "topics")

    def __init__(self, key, keep, protect, act, metric, val, val2, topics):
        self.key = key
        self.keep = keep
        self.protect = protect
        self.act = act
        self.metric = metric
        self.val = val
        self.val2 = val2
        self.topics = tuple(topics)
        if len(self.topics) != 6:
            raise ValueError("domain %s needs 6 topic facets" % key)


def D(key, keep, protect, act, metric, val, val2, topics):
    return Domain(key, keep, protect, act, metric, val, val2, topics)


DOMAINS = [
    D("modbus", "从站地址表", "波特率配置", "重连", "重连间隔", "1.5秒", "2秒",
      ("Modbus重连退避", "Modbus从站地址", "Modbus轮询周期", "Modbus寄存器偏移",
       "Modbus写保护", "Modbus超时保护")),
    D("can", "报文周期表", "终端电阻配置", "仲裁", "位时间", "2微秒", "1.8微秒",
      ("CAN总线仲裁", "CAN报文周期", "CAN采样点", "CAN错误帧", "CAN终端电阻", "CAN过滤器")),
    D("ethercat", "从站配置表", "分布时钟参数", "同步", "同步抖动", "80纳秒", "60纳秒",
      ("EtherCAT同步", "EtherCAT从站", "EtherCAT周期", "EtherCAT诊断",
       "EtherCAT分布时钟", "EtherCAT掉线")),
    D("serial", "帧格式说明", "校验位设置", "握手", "超时", "200毫秒", "150毫秒",
      ("串行链路握手", "串行链路缓冲", "串行链路校验", "串行链路超时", "串行链路粘包",
       "串行链路重传")),
    D("spi", "寄存器映射表", "时钟极性设置", "回读", "时钟频率", "8兆赫", "6兆赫",
      ("SPI时钟极性", "SPI片选时序", "SPI回读校验", "SPI速率切换", "SPI模式号",
       "SPI片选间隔")),
    D("i2c", "设备地址表", "上拉电阻值", "重试", "总线时钟", "100千赫", "400千赫",
      ("I2C总线锁死", "I2C上拉电阻", "I2C地址冲突", "I2C时钟拉伸", "I2C应答缺失",
       "I2C重试次数")),
    D("stepper", "细分设置表", "保持电流值", "加减速", "丢步数", "3步", "1步",
      ("步进丢步现象", "步进细分配置", "步进加减速", "步进堵转保护", "步进保持电流",
       "步进回零")),
    D("servo", "增益参数表", "抱闸配置", "使能", "跟随误差", "12脉冲", "8脉冲",
      ("伺服使能时序", "伺服增益整定", "伺服跟随误差", "伺服抱闸逻辑", "伺服报警复位",
       "伺服输出限幅")),
    D("reducer", "背隙测量记录", "预紧力参数", "回差", "回差量", "4角分", "2角分",
      ("减速器背隙", "减速器回差", "减速器效率", "减速器温升", "减速器预紧", "减速器润滑")),
    D("leadscrew", "预紧记录表", "支撑座垫片", "预紧", "轴向间隙", "0.02mm", "0.01mm",
      ("丝杠预紧力", "丝杠轴向间隙", "丝杠螺距误差", "丝杠支撑刚度", "丝杠润滑周期",
       "丝杠热伸长")),
    D("guide", "平行度记录", "导轨压板", "调平", "平行度", "0.03mm", "0.01mm",
      ("导轨平行度", "导轨直线度", "导轨滑块间隙", "导轨润滑膜", "导轨安装基准",
       "导轨防尘罩")),
    D("gripper", "行程记录表", "气压设定值", "夹持", "闭合时间", "0.4秒", "0.3秒",
      ("夹爪行程限位", "夹爪夹持力", "夹爪闭合时间", "夹爪气路泄漏", "夹爪重复定位",
       "夹爪磨损量")),
    D("conveyor", "速度曲线表", "变频器参数", "调速", "线速度", "0.5米每秒", "0.4米每秒",
      ("传送带调速", "传送带打滑", "传送带张紧", "传送带偏载", "传送带启停", "传送带积放")),
    D("sorting", "节拍记录表", "分拣阈值", "分拣", "单件节拍", "380毫秒", "350毫秒",
      ("分拣节拍", "分拣误判率", "分拣队列积压", "分拣落料时序", "分拣工位切换",
       "分拣缓存容量")),
    D("lidar", "时间戳记录", "外参文件", "对时", "时间偏移", "1.2毫秒", "0.8毫秒",
      ("雷达时间戳", "雷达帧丢失", "雷达点云失真", "雷达转速波动", "雷达回波强度",
       "雷达近距盲区")),
    D("gyro", "温漂记录表", "零偏参数", "补偿", "温漂量", "0.05度", "0.02度",
      ("陀螺仪温漂", "陀螺仪零偏", "陀螺仪噪声", "陀螺仪量程切换", "陀螺仪自检",
       "陀螺仪安装角")),
    D("accel", "零偏记录表", "量程配置", "去偏", "零偏量", "0.3度", "0.1度",
      ("加速度计零偏", "加速度计噪声", "加速度计量程", "加速度计安装面", "加速度计温漂",
       "加速度计自检")),
    D("mag", "补偿记录表", "滤波参数", "拟合", "残差", "2度", "0.5度",
      ("磁力计零偏补偿", "磁力计残差", "磁力计软铁干扰", "磁力计硬铁干扰", "磁力计安装角",
       "磁力计采样率")),
    D("temp", "采样时序表", "采样周期", "采样", "稳定时间", "12秒", "8秒",
      ("温度采样时序", "温度读数漂移", "温度补偿系数", "温度报警阈值", "温度探头接触",
       "温度采集抖动")),
    D("ripple", "纹波记录表", "电感取值", "滤波", "纹波峰峰值", "45毫伏", "30毫伏",
      ("母线纹波", "纹波峰峰值", "电解电容老化", "输入滤波电感", "上电冲击电流", "输出过冲")),
    D("emc", "整改记录表", "接地方式", "整改", "裕量", "6分贝", "9分贝",
      ("传导发射裕量", "辐射发射裕量", "静电放电抗扰", "群脉冲抗扰", "接地环路", "屏蔽层搭接")),
    D("reflow", "温区记录表", "传送速度", "调温", "峰值温度", "245摄氏度", "240摄氏度",
      ("回流焊温区", "回流焊峰值温", "焊点空洞率", "锡膏印刷厚度", "预热斜率", "冷却速率")),
    D("fdm", "层高记录表", "喷嘴温度", "切片", "层高偏差", "0.06mm", "0.04mm",
      ("打印层高偏差", "挤出量偏差", "首层附着", "桥接塌陷", "回抽长度", "打印翘边")),
    D("fan", "曲线记录表", "启动阈值", "控温", "转速波动", "60转", "40转",
      ("风扇转速曲线", "风扇启动阈值", "风扇噪声", "风扇积灰", "风道回流", "散热余量")),
    D("memleak", "占用记录表", "对象池大小", "回收", "常驻内存", "180兆", "120兆",
      ("常驻内存增长", "句柄泄漏", "对象池回收", "长期运行占用", "采样点泄漏", "缓存未释放")),
    D("logrotate", "轮转记录表", "保留天数", "轮转", "单文件大小", "64兆", "32兆",
      ("记录轮转策略", "保留天数", "写入延迟", "分片大小", "压缩比", "落盘顺序")),
    D("watchdog", "喂狗记录表", "超时阈值", "喂狗", "复位次数", "2次", "0次",
      ("看门狗喂狗", "看门狗复位", "看门狗超时阈值", "任务心跳丢失", "异常重启统计",
       "复位原因记录")),
    D("filelock", "加锁顺序表", "锁超时设置", "加锁", "等待时长", "120毫秒", "60毫秒",
      ("文件锁竞争", "锁等待时长", "加锁顺序", "死锁复现", "锁粒度", "锁释放时机")),
    D("diskcache", "写盘记录表", "刷盘周期", "刷盘", "落盘延迟", "15毫秒", "8毫秒",
      ("磁盘写入缓存", "落盘延迟", "缓存丢失风险", "写放大", "顺序写吞吐", "刷盘周期")),
    D("tcpwin", "窗口参数表", "重传阈值", "重传", "重传窗口", "64千字节", "32千字节",
      ("传输重传窗口", "拥塞窗口", "连接复用时序", "握手往返延迟", "接收缓冲不足",
       "慢启动阈值")),
    D("dns", "解析记录表", "缓存时长", "解析", "解析耗时", "18毫秒", "9毫秒",
      ("域名解析缓存", "解析耗时", "解析超时重试", "解析结果漂移", "本地缓存命中",
       "解析并发上限")),
    D("ntp", "对时记录表", "同步间隔", "对时", "时钟偏差", "2毫秒", "0.5毫秒",
      ("时钟同步偏差", "校时间隔", "时钟漂移率", "同步源切换", "闰秒处理", "时间戳单调性")),
    D("autostart", "启动顺序表", "依赖声明文件", "自启", "就绪耗时", "6秒", "4秒",
      ("服务自启顺序", "依赖声明缺失", "启动竞态", "就绪探测", "启动超时回退",
       "启动日志归属")),
    D("image", "层清单", "基础镜像版本", "瘦身", "镜像体积", "820兆", "540兆",
      ("镜像体积瘦身", "镜像层缓存", "镜像构建时长", "容器启动耗时", "镜像基础层版本",
       "镜像漏洞扫描")),
    D("connpool", "连接数记录", "池上限", "复用", "等待队列", "12个", "6个",
      ("连接池上限", "连接复用", "连接泄漏", "等待队列积压", "连接探活", "池预热")),
    D("isolation", "隔离级别说明", "默认级别", "提交", "锁等待", "40毫秒", "20毫秒",
      ("事务隔离级别", "幻读复现", "脏读边界", "间隙锁范围", "提交时序", "回滚代价")),
    D("slowquery", "慢查询记录", "执行计划快照", "优化", "查询耗时", "900毫秒", "250毫秒",
      ("慢查询耗时", "执行计划漂移", "全表扫描", "排序溢出磁盘", "统计信息陈旧",
       "连接顺序选择")),
    D("sharding", "路由规则表", "分片键", "路由", "跨片比例", "8%", "3%",
      ("分片键选择", "路由规则漂移", "跨片查询比例", "分片扩容", "热点分片", "分片再平衡")),
    D("cachebreak", "回源记录表", "预热脚本", "预热", "回源比例", "22%", "6%",
      ("缓存击穿", "缓存雪崩", "热点键预热", "过期时间抖动", "回源比例", "缓存穿透")),
    D("mq", "堆积监控表", "消费并发数", "消费", "堆积量", "4万条", "1万条",
      ("消息堆积", "消费重复", "消费位点回退", "分区倾斜", "死信队列", "消息延迟")),
    D("idempotent", "去重键规则", "键生成方式", "去重", "重复率", "3%", "0.2%",
      ("幂等去重键", "重复提交", "去重窗口", "键冲突概率", "重试语义", "补偿事务")),
    D("ratelimit", "限流记录表", "令牌桶容量", "限流", "拒绝率", "5%", "1%",
      ("接口限流阈值", "令牌桶容量", "突发流量削峰", "限流误伤", "排队等待", "降级开关")),
    D("canary", "灰度名单", "放量步长", "放量", "错误率", "0.8%", "0.2%",
      ("灰度发布比例", "灰度名单命中", "放量步长", "灰度回退条件", "观测指标口径",
       "新旧版本并存")),
    D("drill", "预案清单", "演练窗口", "演练", "切换耗时", "90秒", "45秒",
      ("发布预案演练", "切换耗时", "回退演练", "演练脚本幂等", "演练窗口", "故障注入范围")),
    D("coverage", "覆盖率报告", "门限阈值", "补测", "行覆盖率", "78%", "86%",
      ("单元测试覆盖率", "分支覆盖缺口", "用例重复度", "断言有效性", "覆盖率门限",
       "用例执行时长")),
    D("static", "告警清单", "规则集合", "整改", "高危告警数", "14个", "3个",
      ("静态分析告警", "误报率", "规则集合版本", "空指针告警", "资源释放告警", "告警基线")),
    D("deps", "依赖清单", "锁文件", "升级", "冲突数", "5处", "1处",
      ("依赖版本锁定", "间接依赖冲突", "锁文件漂移", "许可合规", "依赖体积", "升级破坏面")),
    D("buildcache", "缓存命中记录", "构建缓存目录", "增量构建", "命中率", "62%", "88%",
      ("构建缓存命中", "构建时长", "增量构建正确性", "缓存污染", "远程缓存同步",
       "构建可复现性")),
    D("toolchain", "工具链版本表", "目标平台配置", "交叉编译", "产物体积", "1.4兆", "1.1兆",
      ("交叉编译工具链", "工具链版本漂移", "目标平台配置", "链接顺序", "静态库重复符号",
       "产物体积")),
    D("boot", "启动耗时记录", "启动参数", "裁剪", "启动耗时", "4.2秒", "2.8秒",
      ("启动加载耗时", "内核裁剪", "启动脚本顺序", "服务就绪判定", "冷启动缓存",
       "启动阶段并发")),
]

KINDS = ("pending", "failed", "completed", "correction", "constraint", "unlanded")
MULTITURN_KINDS = ("pending", "completed", "unlanded")

# --------------------------------------------------------------------------
# templates.  {topic}/{keep}/{protect}/{act}/{metric}/{val}/{val2} are filled
# from the domain table.  Assertion slots are phrased so that no slot value is a
# verbatim substring of the source turn (DATA.validate's copy gate).
# --------------------------------------------------------------------------
USER_TEXT = {
    "pending": (
        "{topic}还没弄完，先把{keep}备份，别动{protect}。",
        "关于{topic}，现在还没有跑通；{keep}先留一份副本，{protect}不要改。",
        "{topic}卡住了，进度停在复现这一步；{keep}要保留，{protect}先别碰。",
        "今天先处理{topic}：还没有结论，{keep}不能丢，{protect}保持原样。",
    ),
    "failed": (
        "{topic}这块，{act}上次报错了，先看日志再重试，{protect}不要动。",
        "{topic}这一版失败了，日志里{metric}异常，先别改{protect}。",
    ),
    "completed": (
        "{topic}跑通了，{metric}是{val}，{keep}也归档了。",
        "{topic}已经过了，{metric}落在{val}，{keep}归档完成。",
    ),
    "correction": (
        "之前记错了，{topic}的{metric}其实是{val2}，不是{val}。",
        "更正确认一下：{topic}的{metric}是{val2}，先前写的{val}作废。",
    ),
    "constraint": (
        "以后{topic}按{val}执行，{protect}保持不动。",
        "{topic}就按{val}来，别再改{protect}。",
    ),
    "unlanded": (
        "{topic}的限制先记下来，别当成已经生效。",
        "{topic}这条只是记一下，还没落到代码里。",
    ),
}
ASSISTANT_TEXT = {
    "pending": (
        "收到。{act}尚未收敛，我先整理{keep}再排查阻塞点；{protect}保持原状，出结果前不下结论。",
        "明白。{act}没有结论，接下来先归档{keep}复现现象；{protect}不会被改动。",
        "好的。先把{keep}归档，再定位{act}卡住的环节；没有证据前不写完成。",
        "记下了。{act}目前只在方案层面，尚未落地；{protect}维持现状。",
    ),
    "failed": (
        "确认失败：{act}返回错误码，日志只显示{metric}异常，原因还没有定位；{protect}保持不动。",
        "失败了：{act}抛出异常，{metric}偏离预期，定位还没有进展；{protect}先放着。",
    ),
    "completed": (
        "结果已确认：{act}过验收，{metric}读数{val}，{keep}归档完成。",
        "确认完成：{act}验收通过，{metric}为{val}，{keep}已归档。",
    ),
    "correction": (
        "已确认更正：{metric}以{val2}为准，先前的记录作废。",
        "收到更正：{metric}改按{val2}执行，旧的{val}不再采用。",
    ),
    "constraint": (
        "已确认：{metric}锁定为{val}，已经落到配置里；{protect}保持原值。",
        "确认执行：{metric}锁在{val}，落到配置里；{protect}维持原值。",
    ),
    "unlanded": (
        "已记录。{protect}尚未接入主流程，接入前不作可用功能。",
        "记下了。{topic}未生效，等接入后再评估。",
    ),
}
SLOTS = {
    "pending": ("约束：{keep}不得被改动",
                "未完成：先复现{act}现象再核对{keep}",
                "结果：尚未确认{act}可用"),
    "failed": ("约束：{protect}排查期间不改动",
               "未完成：定位{act}报错来源",
               "结果：尚未修复{act}故障"),
    "completed": ("约束：无", "未完成：无",
                  "结果：已确认{act}验收合格，{metric}实测{val}"),
    "correction": ("约束：{metric}只认更正后的取值", "未完成：无",
                   "结果：已确认{metric}更正为{val2}"),
    "constraint": ("约束：{metric}不得偏离{val}", "未完成：无",
                   "结果：已确认{metric}写进配置，取值{val}"),
    "unlanded": ("约束：{protect}的限制接入前不作可用",
                 "未完成：把{protect}的限制接入主流程",
                 "结果：尚未接入，暂不作为可用功能"),
}
HISTORY = {
    "pending": ("【历史轮次】\n用户：{topic}先记一下，{keep}留个副本。\n"
                "助手：记下了，{keep}会保留，{protect}维持原样。\n【当前轮次】\n"),
    "completed": ("【历史轮次】\n用户：{topic}先按{keep}的口径推进。\n"
                  "助手：好的，{protect}不动。\n【当前轮次】\n"),
    "unlanded": ("【历史轮次】\n用户：{topic}先记着，别写成已生效。\n"
                 "助手：已记录，等接入后再看效果。\n【当前轮次】\n"),
}


def fields(domain, topic):
    return {"topic": topic, "keep": domain.keep, "protect": domain.protect,
            "act": domain.act, "metric": domain.metric, "val": domain.val,
            "val2": domain.val2}


def build_case(case_id, group, domain, kind, domain_index, kind_index, multi_turn):
    topic = domain.topics[kind_index]
    values = fields(domain, topic)
    user = USER_TEXT[kind][(domain_index + kind_index) % len(USER_TEXT[kind])].format(**values)
    if multi_turn:
        user = HISTORY[kind].format(**values) + user
    assistant = ASSISTANT_TEXT[kind][
        (domain_index * 3 + kind_index) % len(ASSISTANT_TEXT[kind])].format(**values)
    constraint, pending, result = SLOTS[kind]
    summary = "\n".join(["主题：" + topic, constraint.format(**values),
                         pending.format(**values), result.format(**values)])
    return {"id": case_id, "group": group, "user": user, "assistant": assistant,
            "unverified": kind in ("pending", "failed", "unlanded"),
            "summary": summary, "kind": kind, "synthetic": SYNTHETIC,
            "generator": SCRIPT_VERSION, "domain": domain.key,
            "multi_turn": bool(multi_turn)}


def generate():
    """Train extras first, then heldout extras: stable, deterministic order."""
    cases = []
    plan = [(DATA.TRAIN, DOMAINS[:TRAIN_DOMAINS], "S"),
            (DATA.HELDOUT, DOMAINS[TRAIN_DOMAINS:TRAIN_DOMAINS + HELDOUT_DOMAINS], "V")]
    for group, domains, prefix in plan:
        number = 0
        for index, domain in enumerate(domains):
            for kind_index, kind in enumerate(KINDS):
                number += 1
                cases.append(build_case("%s%03d" % (prefix, number), group, domain, kind,
                                        index, kind_index, kind in MULTITURN_KINDS))
    return cases


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
class ReportError(RuntimeError):
    """A gate the generated dataset must satisfy was violated."""


def longest_common_substring(left, right):
    if not left or not right:
        return ""
    previous = [0] * (len(right) + 1)
    best_length, best_end = 0, 0
    for i, char in enumerate(left, 1):
        current = [0] * (len(right) + 1)
        for j, other in enumerate(right, 1):
            if char == other:
                current[j] = previous[j - 1] + 1
                if current[j] > best_length:
                    best_length, best_end = current[j], i
        previous = current
    return left[best_end - best_length:best_end]


def slot_pairs(summary):
    return dict(DATA._slot_pairs(summary))


def check_copies(cases):
    """Verbatim-copy gate (DATA's rule) plus a shared-run probe."""
    verbatim = []
    shared_runs = []
    runs_at_least_8 = 0
    topic_only = []
    for case in cases:
        source = case["user"] + "\n" + case["assistant"]
        topic = slot_pairs(case["summary"])["topic"]
        for key, value in slot_pairs(case["summary"]).items():
            if key == "topic" or value == DATA.EMPTY_SLOT or len(value) < LCS_LIMIT:
                continue
            if value in source:
                verbatim.append({"case": case["id"], "slot": key, "value": value})
            run = longest_common_substring(value, source)
            if len(run) >= LCS_LIMIT:
                runs_at_least_8 += 1
                without_topic = longest_common_substring(
                    value.replace(topic, ""), source.replace(topic, ""))
                record = {"case": case["id"], "slot": key, "run": run,
                          "len": len(run), "run_without_topic": len(without_topic)}
                shared_runs.append(record)
                if len(without_topic) < LCS_LIMIT:
                    topic_only.append(record)
    if verbatim:
        raise ReportError("verbatim source copies: %s" % json.dumps(verbatim[:3],
                                                                    ensure_ascii=False))
    without_topic_runs = [r["run_without_topic"] for r in shared_runs]
    offenders = {}
    for record in shared_runs:
        if record["run_without_topic"] >= LCS_LIMIT:
            offenders.setdefault(record["run"], []).append(record["case"])
    worst = sorted([{"run": run, "cases": len(ids)} for run, ids in offenders.items()],
                   key=lambda item: (-item["cases"], item["run"]))[:12]
    return {"verbatim_copies": 0, "shared_runs_at_least_8": runs_at_least_8,
            "topic_attributable_runs": len(topic_only),
            "non_topic_runs_at_least_8": runs_at_least_8 - len(topic_only),
            "distinct_offending_runs": len(offenders),
            "worst_offending_runs": worst,
            "longest_run_after_removing_the_topic": max(without_topic_runs, default=0),
            "samples": shared_runs[:5],
            "topic_only_samples": topic_only[:3]}


def check_topics(cases):
    """Topic uniqueness, length, and disjointness from every existing subject."""
    topics = [slot_pairs(c["summary"])["topic"] for c in cases]
    duplicates = sorted({t for t in topics if topics.count(t) > 1})
    if duplicates:
        raise ReportError("duplicate topics: %s" % duplicates[:5])
    too_long = [t for t in topics if len(t) > 14]
    if too_long:
        raise ReportError("topics longer than 14 chars: %s" % too_long[:5])
    existing = {slot_pairs(c["summary"])["topic"] for c in DATA.cases()}
    overlap = sorted(set(topics) & existing)
    if overlap:
        raise ReportError("topics overlap existing cases: %s" % overlap[:5])
    hits = []
    for case in cases:
        blob = case["user"] + "\n" + case["assistant"] + "\n" + case["summary"]
        for term in FORBIDDEN_TERMS:
            if term in blob:
                hits.append({"case": case["id"], "term": term})
    if hits:
        raise ReportError("forbidden subject terms present: %s" % json.dumps(hits[:5],
                                                                            ensure_ascii=False))
    shared = sorted({topic[:2] for topic in topics} &
                    {topic[:2] for topic in HELDOUT_TOPICS} - {""})
    return {"unique_topics": len(set(topics)), "max_topic_chars": max(len(t) for t in topics),
            "overlap_with_existing_topics": 0, "forbidden_term_hits": 0,
            "shared_two_char_prefix_with_H_topics": shared}


def check_slots(cases):
    summaries = [c["summary"] for c in cases]
    if len(set(summaries)) != len(summaries):
        raise ReportError("duplicate memory text")
    longest = max(len(s) for s in summaries)
    if longest > 200:
        raise ReportError("summary longer than 200 chars: %d" % longest)
    distinct = {}
    for key in DATA.SLOT_KEYS:
        values = [slot_pairs(c["summary"])[key] for c in cases]
        distinct[key] = len(set(values))
    repeated = {}
    for key in ("constraint", "pending", "result"):
        counter = {}
        for case in cases:
            value = slot_pairs(case["summary"])[key]
            counter[value] = counter.get(value, 0) + 1
        repeated[key] = sorted(
            [{"value": v, "count": n} for v, n in counter.items() if n > 1],
            key=lambda item: -item["count"])[:3]
    lengths = [len(c["user"]) for c in cases]
    return {"unique_summaries": len(set(summaries)), "max_summary_chars": longest,
            "distinct_slot_values": distinct, "most_repeated_slots": repeated,
            "max_user_chars": max(lengths), "min_user_chars": min(lengths)}


def check_split_ordering(loaded, extras):
    """Report whether the loader's ordering still matches the trainer's slicing.

    train_rwkv_slots.py builds the feature matrix in `items` order and then takes
    train rows as matrix[:train_count] and heldout rows as matrix[train_count:],
    which is only valid when every train case precedes every heldout case.
    `all_cases` used to return embedded cases (16 train, then 6 heldout) and
    APPEND the extras, so any extras file put heldout rows inside the train
    slice; it now sorts by (group, id). This check measures the live behaviour
    instead of assuming either version.
    """
    groups = [c["group"] for c in loaded]
    total_train = sum(1 for g in groups if g == DATA.TRAIN)
    prefix = groups[:total_train]
    misplaced_train = sum(1 for g in groups[total_train:] if g == DATA.TRAIN)
    misplaced_heldout = sum(1 for g in prefix if g == DATA.HELDOUT)
    consistent = misplaced_heldout == 0 and misplaced_train == 0
    ids = [c["id"] for c in loaded]
    extras_train = [c for c in extras if c["group"] == DATA.TRAIN]
    extras_heldout = [c for c in extras if c["group"] == DATA.HELDOUT]
    return {
        "combined_order": ("%d embedded train, %d embedded heldout, then %d extra train, "
                           "%d extra heldout" % (len(DATA.TRAIN_CASES), len(DATA.HELDOUT_CASES),
                                                 len(extras_train), len(extras_heldout))),
        "train_count_used_by_slicing": total_train,
        "heldout_rows_sitting_in_the_train_slice": misplaced_heldout,
        "train_rows_sitting_in_the_heldout_slice": misplaced_train,
        "misplaced_rows_total": misplaced_heldout + misplaced_train,
        "slice_split_consistent": consistent,
        "extras_emitted_in_sorted_id_order": [c["id"] for c in extras] == sorted(c["id"] for c in extras),
        "ordering_note": ("all_cases returns train cases before heldout cases "
                          "(sorted by group then id); the slice-based split in "
                          "train_rwkv_slots.py therefore agrees with the case groups"
                          if consistent else
                          "MISMATCH: all_cases interleaves groups, so matrix[:train_count] "
                          "and matrix[train_count:] do not correspond to the group split"),
        "required_fix": None if consistent else
                        ("train_rwkv_slots.py must index rows by case group "
                         "(train_index = [i for i, c in enumerate(items) "
                         "if c[\"group\"] == DATA.TRAIN]) instead of slicing "
                         "matrix[:train_count]."),
        "ids_in_order_sample": ids[:2] + ["..."] + ids[-2:],
    }


def check_loader(cases_path, expected=None):
    """Load through the real loader. A stray template.json would shadow --cases."""
    if NO_TEMPLATE.exists():
        raise ReportError("refusing to run: %s exists" % NO_TEMPLATE)
    extras = DATA.load_events(cases_path=cases_path, template_path=NO_TEMPLATE)
    if not extras:
        raise ReportError("loader returned no extra cases from %s" % cases_path)
    if expected is not None and len(extras) != len(expected):
        raise ReportError("loader returned %d cases, file has %d"
                          % (len(extras), len(expected)))
    for case in extras:
        for field in ("synthetic", "kind", "generator", "multi_turn", "domain"):
            if field in case:
                raise ReportError("load_events kept %s; the loader contract changed" % field)
    loaded = DATA.all_cases(cases_path=cases_path, template_path=NO_TEMPLATE)
    embedded_ids = {c["id"] for c in DATA.cases()}
    collisions = sorted({c["id"] for c in extras} & embedded_ids)
    if collisions:
        raise ReportError("extra ids collide with embedded ids: %s" % collisions[:5])
    return extras, loaded


def check_truthfulness(cases):
    """Every measurement the summary claims must actually appear in its source.

    Synthetic or not, a label may not invent a number: if 结果 says 实测1.5秒,
    the event must contain 1.5秒. Digits in a summary can only come from the
    measurement tokens, so this is a cheap and complete check.
    """
    invented = []
    checked = 0
    for case in cases:
        source = case["user"] + "\n" + case["assistant"]
        for token in re.findall(r"[0-9][0-9.]*", case["summary"]):
            checked += 1
            if token not in source:
                invented.append({"case": case["id"], "token": token})
    if invented:
        raise ReportError("summary numbers absent from the source: %s"
                          % json.dumps(invented[:5], ensure_ascii=False))
    return {"numeric_tokens_checked": checked, "invented_measurements": 0}


def build_report(cases_path, meta_path, extras, loaded, marks):
    kinds = {}
    multi_turn = 0
    for case in extras:
        mark = marks.get(case["id"], {})
        if mark.get("kind"):
            kinds[mark["kind"]] = kinds.get(mark["kind"], 0) + 1
        multi_turn += 1 if mark.get("multi_turn") else 0
    counts = DATA.split_counts(loaded)
    extras_counts = DATA.split_counts(extras)
    report = {
        "ok": True,
        "synthetic": SYNTHETIC,
        "label": LABEL,
        "generator": SCRIPT_VERSION,
        "interpreter": sys.version.split()[0],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": {"cases": str(cases_path), "meta": str(meta_path)},
        "counts": {
            "extras_total": len(extras),
            "extras_train": extras_counts["train"],
            "extras_heldout": extras_counts["heldout"],
            "combined_total": counts["total"],
            "combined_train": counts["train"],
            "combined_heldout": counts["heldout"],
            "by_kind": kinds,
            "unverified_extras": sum(1 for c in extras if c["unverified"]),
            "verified_extras": sum(1 for c in extras if not c["unverified"]),
            "multi_turn_extras": multi_turn,
            "domains": len({mark.get("domain") for mark in marks.values() if mark.get("domain")}),
            "documented_as_synthetic_in_file": all(
                marks.get(case["id"], {}).get("synthetic") is True for case in extras),
        },
        "gates": {
            "data_validate": "pass (DATA.all_cases ran validate on the combined set)",
            "id_collisions_with_embedded": 0,
            "notes": "non-topic slot values >=8 chars are re-checked against the source",
        },
        "split_ordering": check_split_ordering(loaded, extras),
        "limitations": [
            "SYNTHETIC: template-generated wording over invented domains; not real events.",
            "Train and heldout extras share kind templates (only the domains differ), so "
            "synthetic heldout is weaker evidence than the embedded real-topic heldout.",
            "Multi-turn history is text inside the user field: the frozen backbone reads "
            "one concatenated string, so this does NOT test recurrent multi-turn memory.",
            "Per-case synthetic/kind/generator fields are dropped by "
            "rwkv_slots_data._normalized; they are file-level provenance only.",
            "Synthetic cases raise the case count, not the evidential strength: a decoder "
            "can fit these templates without reading the recurrent state.",
        ],
    }
    report["gates"].update(check_slots(extras))
    report["gates"].update(check_topics(extras))
    report["gates"].update(check_truthfulness(extras))
    report["copy_probe"] = check_copies(extras)
    return report


def provenance(cases_path):
    """File-level provenance per id. `load_events` drops every extra field, so the
    kind/multi-turn/domain tallies must come from the raw JSONL lines."""
    found = {}
    text = Path(cases_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        found[raw["id"]] = {"kind": raw.get("kind"), "multi_turn": bool(raw.get("multi_turn")),
                            "domain": raw.get("domain"), "synthetic": raw.get("synthetic")}
    return found


def write_cases(path, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate SYNTHETIC extra cases (v2) for the RWKV-state pilot.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--check-only", action="store_true",
                        help="validate the existing cases file without regenerating")
    parser.add_argument("--quiet", action="store_true", help="print only the key counts")
    args = parser.parse_args(argv)
    cases_path = Path(args.out)
    meta_path = Path(args.meta)
    try:
        if args.check_only:
            extras, loaded = check_loader(cases_path)
            cases = []
        else:
            cases = generate()
            write_cases(cases_path, cases)
            extras, loaded = check_loader(cases_path, expected=cases)
        report = build_report(cases_path, meta_path, extras, loaded,
                              provenance(cases_path))
        report["mode"] = "check-only" if args.check_only else "generate"
        report["outputs"]["bytes"] = cases_path.stat().st_size
        report["outputs"]["sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
        report["outputs"]["combined_data_hash"] = DATA.data_version(loaded)
        write_json(meta_path, report)
    except (ReportError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2
    if args.quiet:
        print(json.dumps({"ok": True, "counts": report["counts"],
                          "gates": {k: v for k, v in report["gates"].items()
                                    if k in ("max_summary_chars", "max_topic_chars",
                                             "max_user_chars", "unique_summaries",
                                             "forbidden_term_hits", "overlap_with_existing_topics")},
                          "copy_probe": report["copy_probe"],
                          "split_ordering": report["split_ordering"]},
                         ensure_ascii=False, indent=2), flush=True)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
