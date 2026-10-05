# Quant推进研究字段参考（v35）

本页配合[模型二指令](02-quant.md)，定义第一步推进与重建研究指标；生产尚未接入评分，独立计算见[评分试算参考](02-quant-score-trial-ref.md)。

本页v35参数和排序用于生产辅助证据。隔离research_v7通过可选候选选择器改为60条搜索、放量资格及最近有效低点排序，详见评分试算参考；未传选择器的原输出保持兼容，不将隔离规则视为生产现行规则。

## 参数与候选选择

配置入口为 `vcp.impulse_evidence`：`enabled`、`lookback_days=30`、`diagnostic_lookback_days=[10,20,30]`、`baseline_volume_days=20`、`min_gain_pct=12`、`min_up_peak_volume_ratio=1.5`、`upper_shadow_warning_ratio=0.5`。已删除`max_bridge_drawdown_pct`；这些是研究观察口径，未用于交易过滤。

设研究阶段首段起点索引S、已确认旧推进耗尽段结束索引F（初始阶段无F）。扫描B从`max(0,S-W,F)`至`S-1`；每个B以`(B,S]`内第一次出现的最高收盘为H。仅`close(H)>close(B)>0`构成正向候选。桥接区`[H,S]`最低收盘与H的绝对回撤始终输出，不再以12%排除。桥接最低收盘不高于B时，候选成果已经全部回吐，作为耗尽背景保存，不冒充仍有效的推进。

先选尚未耗尽且涨幅达到`min_gain_pct`的候选；其次选尚未耗尽弱候选；最后仅保留耗尽候选。各层按H索引降序、涨幅降序、B索引降序选择，表达最近一轮实质推进及同高点的完整起点。无正向候选为`NO_ADVANCE`。`candidate_summaries`保留全部扫描候选的起止、涨幅、桥接回撤/保留及是否耗尽；`latest_weak_candidate`保留最近弱候选，`strongest_exhausted_candidate`另存最大涨幅的耗尽背景，二者不会自动取代主参照。窗口边界起点标记`BASE_AT_SCAN_BOUNDARY`；由重建边界限制则标记`BASE_AT_REBUILD_BOUNDARY`，二者含义不同。

锚点选择只读取`df[:S+1]`。相同数据、窗口、S及已知重建边界下，追加未来回撤/反弹不改变该阶段B/H；可以新增研究阶段，但不能改写旧阶段。公司行为按目标日点时前复权；阶段ID和重建原因明确保留。

## 耗尽与重建研究阶段

按原收缩段顺序观察，研究阶段包含原段编号`group_round_numbers`。当原段为`CONFIRMED`且结束收盘不高于该阶段B时，记录`exhaustion`：段内首次收盘不高于B的事实日期、段结束日期及结束收盘、原右侧确认日期（结束索引加`required_right_confirm_days`）、原段编号。仅日内刺破或尚未确认段不建立重建分界。

这里的确认日期是研究计算复用原段的右侧价格确认边界，不是原系统历史首次选中该段的时间。以目标日原组选组为输入的前缀检查证明阶段锚点不使用其起点之后价格，不能据此声称该组选组在更早日期已被系统发现。

只有确认日期不晚于下一原段开始日，才可从耗尽段结束及之后建立新的推进参照；确认条数缺失、非有限值或不是非负整数时标记`CONFIRMATION_TIMING_UNAVAILABLE`，不可臆定提前可得。确认日期尚未到达时也不重建。新参照达到价格推进标签为`OPEN`，不足或没有推进为`REBUILD_PENDING`，该状态不代表正式交易重建已确认。再次耗尽可以产生后续阶段。

旧阶段`observed_through`为下一阶段开始前一条或分析日，所有路径在该范围内计算。每阶段保留独立B/H、原段编号、原确认状态及原始保留率；待重建阶段不替旧推进抹去失败记录。生产原组及其评分不变。

## 对象与状态

顶层 `impulse_evidence`：

- `schema=quant_impulse_evidence_v2`、`research_only=true`、`as_of`、`group_start_date`（当前研究阶段）、`original_group_start_date`、`lookback_days`、`status`、`reasons`。
- `selected`：配置主窗口的完整结果。
- `windows`：窗口条数为键的完整结果；各窗口`episodes`保留完整研究阶段，`active_episode_id`说明当前参照，顶层窗口字段兼容读取当前阶段指标。

顶层特殊状态：`DISABLED`、`DATA_ISSUE`、`NO_GROUP`、`GROUP_ANCHOR_MISMATCH`。窗口状态：

| 状态 | 含义 |
|---|---|
| IDENTIFIED | 有关联的正向推进，涨幅至少12%；放量是否确认单列 |
| WEAK_ADVANCE | 有关联正向推进，涨幅不足12% |
| EXHAUSTED | 首段开始之前所选候选已经全部回吐，仅保留历史背景 |
| NO_ADVANCE | 当前窗口未找到正向候选；不表示全历史没有推进 |
| DATA_ISSUE | 价格/日期/区间缺失、冲突或非正数，不能计算 |

`selected.anchor`含B/H日期、索引、收盘、主窗口边界及 `anchor_id`（窗口与日期组成，不含会因复权变化的价格）。`selected.price`含涨幅、交易记录间隔、H距S间隔、桥接最低/回撤及效率。所有百分数带 `_pct`，效率和影线比例为0—1。

## 量价公式

- `gain_pct=(H/B-1)*100`。
- `path_efficiency=(H-B)/sum(abs(close[t]-close[t-1]))`，t范围 `(B,H]`；正向推进取值0—1。
- 推进成交量区间 `(B,H]`，按每条收盘相对前一条收盘上涨、下跌、持平分类。基准取B之前最多20条，实际条数另列；不足20条标记 `BASELINE_INCOMPLETE`，放量确认留空。
- 量能口径 `volume_policy=exclude_one_price_up_v1`：日线OHLC有效、严格相等且收盘高于前收时识别为一字上涨日，从推进及基准量样本中同时剔除成交量和样本天数。这是成交受限形态代理，不声称核验了交易所涨停价；不按统一10%阈值识别，不排除普通缩量上涨、持平或一字下跌。缺少OHLC时不猜测剔除，并标记 `ONE_PRICE_CHECK_INCOMPLETE`。
- `baseline_days`、`advance_days`仍是原窗口条数；`baseline_effective_days`、`advance_effective_days`为量能有效样本数，`baseline_excluded_dates`、`advance_excluded_dates`记录剔除日期；方向天数与峰量、占比均按剩余样本计算。原窗口不向前扩展；价格涨幅、路径效率和耗时完全保留一字日。全被剔除时标记 `VOLUME_SAMPLE_EMPTY`及`VOLUME_DATA_INCOMPLETE`，比值留空。原始缺失或非正成交量仍报数据不足，不借剔除掩盖缺失。
- 分别输出推进均量、上涨/下跌日均量和峰量相对基准倍数，以及上涨日量占整个推进段量的比例。 `up_peak_volume_confirmed` 只在基准完整且上涨日峰量倍数达到1.5时为真；不决定价格状态。
- 最大量日附日期、整日收盘涨跌和上影/振幅。最大量在下跌日标记 `PEAK_VOLUME_ON_DOWN_DAY`；上影比例至少0.5标记 `PEAK_VOLUME_LONG_UPPER_SHADOW`。这些是证据警示，不作结构排除。
- 价格有效、但量缺失或非正数时，量证据标记 `VOLUME_DATA_INCOMPLETE`，不制造0倍或已确认放量；收盘仍可用于保留研究。无法使用的日内高低另列 `INTRADAY_DATA_INCOMPLETE`。

## 保留公式与路径

固定B/H，任意价格L的原始保留率为 `(L-B)/(H-B)*100`，不截断：低于B可为负、高于H可超过100。这是原始研究值，试算评分另按当前阶段收缩段最低收盘计算并截断系数，见[评分试算参考](02-quant-score-trial-ref.md)。

- `retention.rounds`仅读取分配给该研究阶段的原段，逐段输出实际最低收盘及日内最低、日期、两种保留率、原回撤和确认状态；不改写原段。
- `retention.post_peak`读取H之后下一条至`observed_through`最低收盘、日内最低和原始保留率，包含桥接及阶段内回撤。H以收盘为锚，不把H当天收盘之前的日内最低当成后续回撤。
- `retention.post_group`读取S到`observed_through`的同类路径，用来区分该阶段首轮开始后的回吐。
- `retention.current_close_retention_pct`计算`observed_through`时的恢复；当前研究阶段该日期就是分析日，已关闭旧阶段截止其观察边界。最低路径始终保留，不用恢复覆盖。

各原收缩段及S起始日的日内最低描述该日线区间的风险范围，不推断起始日内部的价格先后。

`lifecycle`单独取`OPEN`、`EXHAUSTED`或`REBUILD_PENDING`，后两者不会改变生产阶段。候选价格状态与阶段生命周期不同，例如历史上识别到的推进保持`IDENTIFIED`，后来耗尽在生命周期与路径中表达。缺失字段用null和原因表达，不用后续反弹补造确认。

## 隔离审计

`scripts/audit_impulse_evidence.py --dates 260915,260916 --output-dir <独立目录>` 从本地只读权威记录补计算字段；不运行生产工作流、不取数、不落生产库。默认比较Git基线4509413的 `screen()` 与当前 `screen()` 在相同日线输入上的全部原字段（排除新增字段），并对原组单独计算新指标。历史记录与当前基线重扫可能有已有版本差别，应单列，不归因于新指标。调用从本地运行实例进行。

附加`--compare-report <旧隔离目录>`对两版研究字段逐条比较，先要求股票范围、原权威记录及旧生产判定指纹完全一致，再输出`comparison_*.csv`和变更数量；主推进起止、研究起点及状态改变分别统计，全部变更不自动解释为识别质量改善。逐窗口核验原段不丢失/重复、阶段锚点有序、保留不跨阶段和右侧确认边界不提前使用。

输出目录必须是`reports/research/`下的新目录，禁止覆盖既有产物。仅新增指标的数据边界处理发生修改时，可用`--verify-report <原隔离目录>`从只读行情重新核算全部研究对象，要求与原输出完全一致，并验证原评分脚本/配置哈希及旧字段比较指纹；另存`final_verification.json`，不改原审计证据。若原字段路径或配置发生变化，必须重新完整回放，不能用该复核替代。

仅补充备选明细时，可加`--finalize-candidates`：允许新增`candidate_summaries`和`latest_weak_candidate`两个字段，其余研究字段必须逐值等于首次完整回放；原评分脚本及配置哈希也必须相同。保存新的`final_evidence_*.json`、`final_fields_*.csv`与`final_verification.json`，不覆盖首次证据。最终对象全部从只读行情复算，并复查阶段前缀、耗尽确认边界和旧判定指纹；完整生产判定比较沿用首次回放，不声称重新执行了生产判定。若核心研究字段变化，拒绝此模式，须另开目录完整回放。
