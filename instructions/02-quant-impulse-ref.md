# Quant推进研究字段参考（v33）

本页配合[模型二指令](02-quant.md)，仅定义第一轮研究指标；评分、权重和保留衰减曲线尚未实现。

## 参数与候选选择

配置入口为 `vcp.impulse_evidence`：`enabled`、`lookback_days=30`、`diagnostic_lookback_days=[10,20,30]`、`baseline_volume_days=20`、`min_gain_pct=12`、`min_up_peak_volume_ratio=1.5`、`max_bridge_drawdown_pct=12`、`upper_shadow_warning_ratio=0.5`。这些是研究观察口径，未用于交易过滤。

设首段起点索引S。对每个窗口W，扫描B索引从 `max(0,S-W)` 到 `S-1`；每个B以 `(B,S]` 内第一次出现的最高收盘为H。仅 `close(H)>close(B)>0` 构成正向候选。平台桥接区为 `[H,S]`，其最低收盘/H收盘回撤不能超过12%；该条件用于排除已大幅跌离的平台背景，不检查S以后的回撤。不能通过不断更换后续低点让不相关的旧推进获得保留优势。

关联候选按 `gain_pct` 降序、B索引降序选择。无合格关联候选时仍保留窗口内最大正向候选为 `UNLINKED`，但不给已关联推进状态；无正向候选为 `NO_ADVANCE`。边界起点标记 `BASE_AT_SCAN_BOUNDARY`，说明起点可能被窗口截断，不能称为完整启动已确认。每个窗口保留候选数量和最强的关联失败候选，供核验是否旧推进与平台断开。

锚点选择只读取 `df[:S+1]`，不读取后续各轮低点。相同数据口径、窗口及S下，追加未来回撤/反弹不改变B/H；公司行为以目标日点时前复权统一价格，索引以原组日期核对。S或窗口变化属于不同研究锚点，必须显式保留，不能悄悄互换。

## 对象与状态

顶层 `impulse_evidence`：

- `schema=quant_impulse_evidence_v1`、`research_only=true`、`as_of`、`group_start_date`、`lookback_days`、`status`、`reasons`。
- `selected`：配置主窗口的完整结果。
- `windows`：窗口条数为键的完整结果，包含候选、选中锚点、量价证据和保留路径。

顶层特殊状态：`DISABLED`、`DATA_ISSUE`、`NO_GROUP`、`GROUP_ANCHOR_MISMATCH`。窗口状态：

| 状态 | 含义 |
|---|---|
| IDENTIFIED | 有关联的正向推进，涨幅至少12%；放量是否确认单列 |
| WEAK_ADVANCE | 有关联正向推进，涨幅不足12% |
| UNLINKED | 存在正向上涨，但平台桥接超过研究回撤上限 |
| NO_ADVANCE | 当前窗口未找到正向候选；不表示全历史没有推进 |
| DATA_ISSUE | 价格/日期/区间缺失、冲突或非正数，不能计算 |

`selected.anchor`含B/H日期、索引、收盘、主窗口边界及 `anchor_id`（窗口与日期组成，不含会因复权变化的价格）。`selected.price`含涨幅、交易记录间隔、H距S间隔、桥接最低/回撤及效率。所有百分数带 `_pct`，效率和影线比例为0—1。

## 量价公式

- `gain_pct=(H/B-1)*100`。
- `path_efficiency=(H-B)/sum(abs(close[t]-close[t-1]))`，t范围 `(B,H]`；正向推进取值0—1。
- 推进成交量区间 `(B,H]`，按每条收盘相对前一条收盘上涨、下跌、持平分类。基准取B之前最多20条，实际条数另列；不足20条标记 `BASELINE_INCOMPLETE`，放量确认留空。
- 分别输出推进均量、上涨/下跌日均量和峰量相对基准倍数，以及上涨日量占整个推进段量的比例。 `up_peak_volume_confirmed` 只在基准完整且上涨日峰量倍数达到1.5时为真；不决定价格状态。
- 最大量日附日期、整日收盘涨跌和上影/振幅。最大量在下跌日标记 `PEAK_VOLUME_ON_DOWN_DAY`；上影比例至少0.5标记 `PEAK_VOLUME_LONG_UPPER_SHADOW`。这些是证据警示，不作结构排除。
- 价格有效、但量缺失或非正数时，量证据标记 `VOLUME_DATA_INCOMPLETE`，不制造0倍或已确认放量；收盘仍可用于保留研究。无法使用的日内高低另列 `INTRADAY_DATA_INCOMPLETE`。

## 保留公式与路径

固定B/H，任意价格L的原始保留率为 `(L-B)/(H-B)*100`，不截断：低于B可为负、高于H可超过100。这不是已定义的评分系数。

- `retention.rounds`按原组逐段读取开始日至结束日实际最低收盘及日内最低，输出日期、值、两种保留率、原段收盘回撤和原确认状态；不改写原段。
- `retention.post_peak`读取H之后下一条至分析日最低收盘、日内最低和原始保留率，包含平台桥接及所有后续回撤。H以收盘为锚，因此不把H当天收盘之前的日内最低当成推进结束后的回撤。
- `retention.post_group`读取S到分析日的同类路径，用来区分首轮开始后的回吐。
- `retention.current_close_retention_pct`单独计算当前恢复；最低路径始终保留，不能用当前价格覆盖。

各原收缩段及S起始日的日内最低描述该日线区间的风险范围，不推断起始日内部的价格先后。

若旧推进无法关联（UNLINKED），仍输出其原始证据及路径，但状态明确未建立当前母结构的关联。任何缺失字段用null和原因表达，不用后续反弹或更长窗口自动补造已确认结论。

## 隔离审计

`scripts/audit_impulse_evidence.py --dates 260915,260916 --output-dir <独立目录>` 从本地只读权威记录补计算字段；不运行生产工作流、不取数、不落生产库。默认比较Git基线4509413的 `screen()` 与当前 `screen()` 在相同日线输入上的全部原字段（排除新增字段），并对原组单独计算新指标。历史记录与当前基线重扫可能有已有版本差别，应单列，不归因于新指标。调用从本地运行实例进行。

输出目录必须是`reports/research/`下的新目录，禁止覆盖既有产物。仅新增指标的数据边界处理发生修改时，可用`--verify-report <原隔离目录>`从只读行情重新核算全部研究对象，要求与原输出完全一致，并验证原评分脚本/配置哈希及旧字段比较指纹；另存`final_verification.json`，不改原审计证据。若原字段路径或配置发生变化，必须重新完整回放，不能用该复核替代。
