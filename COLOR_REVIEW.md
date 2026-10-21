# apt_0 颜色标注核对

2026-10-17：逐个检查 9 个导航目标的双视角 Habitat 渲染画面，填写 annotations.json。
检查 GLB 材质后确认这些资产使用 baseColorTexture，因此没有把材质名称或缺省颜色因子当作颜色真值。
标注依据是实际渲染外观，不是数据集原生颜色 GT；颜色为大致类别，会受光照、阴影和显示设备影响。

| 物体 ID | 主色 | 核对图片 |
|---|---|---|
| frl_apartment_chair_04_:0000 | 白色 | outputs/color_review/00.jpg |
| frl_apartment_chair_04_:0001 | 白色 | outputs/color_review/01.jpg |
| frl_apartment_chair_05_:0000 | 棕色 | outputs/color_review/02.jpg |
| frl_apartment_chair_05_:0001 | 棕色 | outputs/color_review/03.jpg |
| frl_apartment_sofa_:0000 | 灰色 | outputs/color_review/04.jpg |
| frl_apartment_stool_02_:0000 | 浅木色凳面、深棕支架 | outputs/color_review/05.jpg |
| frl_apartment_table_01_:0000 | 深棕色 | outputs/color_review/06.jpg |
| frl_apartment_table_02_:0000 | 浅木色 | outputs/color_review/07.jpg |
| frl_apartment_table_03_:0000 | 米白色（浅灰白） | outputs/color_review/08.jpg |

部件差异和邻近物体混淆写在 color_description 中。同色椅子各有两把，建议指令为“最近的白色椅子”。
只标注当前 apt_0 的上述实例；切换场景需重新核对，不应假定实例 ID 能跨场景表达同一物体。
服务在初始化时读入标注，修改后需重启。既有会话目录的 scene.json 不会自动重写。

复核渲染：`/home/caosiqi/miniconda3/envs/habitat/bin/python humanoid_nav_demo/check_colors.py`。
