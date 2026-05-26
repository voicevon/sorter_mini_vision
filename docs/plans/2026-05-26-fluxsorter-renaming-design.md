# 命名重构设计方案：从 Mini 演进为 FluxSorter

> **本文档详细定义了将整个分拣系统及子系统的命名从历史遗留的 `mini` 前缀彻底重构为 `FluxSorter`（流光分拣）体系的实施蓝图。**
> 所有涉及的目录重命名、代码文本替换、BLE 协议参数修改均以此设计为准。

---

## 1. 物理重命名设计 (Directory Renaming)

为了使物理目录名称与新的系统设计相匹配，我们将对工作空间内的主要子系统文件夹进行统一重命名。重命名后以 `flux_sorter` 作为统一的前缀：

| 现有文件夹名称 | 重命名后文件夹名称 | 平台 / 说明 |
| :--- | :--- | :--- |
| `sorter_mini_vision` | `flux_sorter_vision` | **Laptop / Rpi / PC 平台**：视觉识别子系统主目录 |
| `sorter_mini_controller` | `flux_sorter_controller` | **ESP32 平台**：分拣控制器固件目录 |
| `sorter_mini_phone` | `flux_sorter_phone` | **Android 平台**：手机端原生分级应用目录 |

---

## 2. 软件代码与协议更新设计 (Code & Protocol Refactoring)

### 2.1 蓝牙 BLE 广播名修改
- **控制器端**：
  在 `sorter_mini_controller/src/BleManager.cpp` 的 `BleManager::begin()` 中，将广播名前缀从 `Sorter_` 变更为 `FluxSorter_`（与系统统一命名对应）。
  - 修改前：`Sorter_%02X%02X%02X`
  - 修改后：`FluxSorter_%02X%02X%02X`
- **视觉端**：
  更新 BLE 扫描与连接逻辑（如 `comm/ble_client.py`），将过滤匹配的前缀从 `Sorter_` 变更为 `FluxSorter_`，确保能正确发现并连接重命名后的分拣控制器。

### 2.2 控制器串口调试日志更新
- 在 `sorter_mini_controller/src/main.cpp` 中，更新串口打印的启动 Banner 信息：
  - 修改前：`LOG_I("--- 分拣机迷你控制器 ---");`
  - 修改后：`LOG_I("--- FluxSorter 控制器 ---");`

### 2.3 文档术语全局替换
- 更新以下文档，将其中提及的 `mini_vision`、`mini_controller` 替换为符合统一命名规则的 `flux_sorter_vision`、`flux_sorter_controller`（手机端对应 `flux_sorter_phone`）：
  - `docs/top_level_overview.md` (顶层全局架构)
  - `docs/design_conclusion.md` (视觉设计结论)
  - `docs/architecture.md` (视觉软件架构)
  - 控制器与视觉系统的各 `README.md`。

---

## 3. 验证方案 (Verification Plan)

### 3.1 编译正确性验证
- **控制器端**：
  在物理重命名为 `flux_sorter_controller` 后，运行 PlatformIO 编译命令，确保其所有本地头文件及多 APP 诊断逻辑能正确构建，无构建路径损坏。
  ```powershell
  & "d:\Software\antigravity\.agent\skills\platformio_compile\scripts\compile_only.ps1"
  ```

### 3.2 扫描逻辑验证
- 在物理重命名为 `flux_sorter_vision` 后，运行视觉端核心代码及测试脚本，确保 Python 端导入路径无误。
