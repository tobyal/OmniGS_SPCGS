# IRIS 工业类别分析

已由 Codex 原生图像查看实际检查 56/56 张训练图：每张联系表含完整 ERP 与六个100°透视面；另检查6个60°局部放大图。脚本只做投影、整理和完整性校验，不执行识别。未查看测试标注来设计词表，未使用分割掩膜监督。

本地 IRIS 元数据有748个CAD实例名称，但共享16个类别ID；27张训练ERP有对应cubemap元数据。名称映射是语义辅助，不是逐像素真值。RMX位姿、56/6/16划分与2048×1024图像保持不变。

选择20类：将泵和电机合为可见机组；将控制柜、接线箱及功能不确定的柜式机械外壳合为设备柜与外壳，避免泛machine与泵重叠。柱、支架、护栏、梯、桥架依据结构分开。新增容器、仪表、灯具和袋垛。色彩是辅助描述，不单独决定类别。

局部复核修正：16号最初疑似板式换热器的矩形物是带敲落孔的接线箱背面；52至56号银筒连接保温管，应归容器候选而非风道。原始初判保留在authored记录，最终独立JSON包含修正。柜式外壳工艺、罐体用途、暗部机械细分仍不确定。

| ID | 类别 | 出现图数 | 可信程度 | 文本依据/区别 |
|---|---|---:|---|---|
|0|pipe / 管道及普通管件|56|high|连续圆管与弯头；灰白保温壳和接缝；与相邻管段连通；易混：duct, tank, column, valve|
|1|valve / 阀门与执行组件|44|high|轮辐或短扳柄；阀杆接入管路；蓝色阀箱及橙执行器；易混：pipe, gauge, pump_motor|
|2|pump motor assembly / 泵电机组|51|high|黑色竖向散热肋；圆机体和金属轴颈；与管路连接但不是仅凭连接判定；易混：valve, cabinet, pipe|
|3|pipe gauge and thermometer / 管路仪表|23|medium|圆盘、指针或刻度；细长黄色刻度壳；接细支管或管壁；易混：valve, pipe, cabinet|
|4|closed cylindrical vessel / 封闭圆柱容器|31|high|封闭顶盖；支脚或独立筒体；蓝漆或银色金属表面；易混：column, pipe, duct|
|5|equipment cabinet and enclosure / 设备柜与箱式外壳|56|medium|薄板矩形外壳；黑门把手或面板开关；小插座箱有蓝红接头；易混：door, wall, pump_motor|
|6|cable tray / 电缆桥架|56|high|开放网篮；内部平行线缆；沿顶及设备上方布置；易混：grating, support, pipe|
|7|ventilation duct / 通风风道|54|high|银色分节薄板；波纹圆段或矩形段；通风口与风道连续；易混：pipe, tank, cabinet|
|8|structural column / 建筑立柱|53|high|粗实心感圆柱；贯通地顶；无设备接头；易混：tank, pipe, wall, support|
|9|metal support frame / 金属支架|56|high|细长打孔钢材；底板螺栓；支撑连接；易混：column, guardrail, ladder, cable_tray|
|10|access ladder / 检修爬梯|17|high|规则横梯级；竖侧梁；环形安全笼；易混：guardrail, support, stairs|
|11|stairway / 楼梯台阶|21|high|连续宽踏面；通往入口；灰色台阶；易混：floor, ladder, grating|
|12|guardrail / 护栏扶手|43|high|横栏或斜扶手；规则立柱；沿人行/设备边缘；易混：pipe, support, ladder|
|13|maintenance platform grating / 检修平台格栅|12|medium|平坦承人平台；粗网格踏面；关联检修梯；易混：cable_tray, stairs, support|
|14|floor / 地面及土建基座|56|high|水平灰面；磨损或污渍；混凝土设备基础；易混：stairs, wall, pump_motor|
|15|wall / 墙面|55|high|连续平面；下红上黑常见；建筑围护；易混：column, cabinet, door|
|16|ceiling / 天花板|55|high|黑色顶部背景；位于设施后方；大面积连续表面；易混：wall, pipe, cable_tray|
|17|building door / 建筑门|12|high|人行尺度开口；门框；关联楼梯入口；易混：cabinet, wall|
|18|strip light / 条形灯具|55|high|细长自发光；白色灯壳；有横向及竖向安装；易混：pipe, duct|
|19|stacked material sacks / 袋装物料|16|high|蓝白袋面；柔软折边；重复叠放；易混：cabinet, floor|

每类描述组合标准对象名、几何、材料、典型外观及位置关系，完整英文prompt和逐图证据见industrial_classes.json。类频率仅表示图像中观察到，不代表像素占比或独立实例数。

资料：[IRIS/IRIS-VIS官方说明](https://centreborelli.github.io/iris-vis/)、[数据集 DOI](https://doi.org/10.5281/zenodo.13777859)、[WACV2025论文](https://openaccess.thecvf.com/content/WACV2025/papers/Armangeon_IRIS-VIS_A_New_Dataset_for_Visibility_Estimation_in_an_Industrial_WACV_2025_paper.pdf)。该工业场景包含复杂遮挡，官方数据提供LiDAR与CAD支持可见性研究。本研究只利用类别元数据帮助命名。