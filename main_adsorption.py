import configparser
import math
import os
import re
import shutil
import threading
import time
from queue import Queue
from threading import Lock
import pandas as pd
import numpy as np
import subprocess # 用于执行外部RASPA命令

# --- RASPA 输出数据解析类 ---
class RASPA_Output_Data:
    '''
    RASPA输出文件对象，用于解析RASPA的.data文件内容。
    '''
    def __init__(self, output_string):
        '''
        初始化时传入RASPA输出文件的字符串
        '''
        self.output_string = output_string
        self.components = re.findall(
            r'Component \d+ \[(.*)\] \(Adsorbate molecule\)', self.output_string)

    def get_components(self):
        return self.components

    def is_finished(self):
        '''
        返回该任务是否已完成
        '''
        pattern = r'Simulation finished'
        result = re.findall(pattern, self.output_string)
        return len(result) > 0

    def get_warnings(self):
        '''
        返回存储警告信息的列表
        '''
        if len(re.findall(r'0 warnings', self.output_string)) > 0:
            return []
        pattern = r'WARNING: (.*)\n'
        return list(set(re.findall(pattern, self.output_string)))

    def get_pressure(self):
        '''
        返回压力，单位是Pa
        '''
        pattern = r'Pressure:\s+(.*)\s+\[Pa\]'
        result = re.findall(pattern, self.output_string)
        return result[0] if result else None # 处理未找到的情况

    def get_excess_adsorption(self, unit='cm^3/g'):
        '''
        指定单位，返回超额吸附量，返回值是一个字典，键是吸附质的名称，值是吸附量
        若不指定单位，默认为cm^3/g
        unit: 'mol/uc','cm^3/g','mol/kg','mg/g','cm^3/cm^3'
        '''
        patterns = {'mol/uc': r"Average loading excess \[molecules/unit cell\]\s+(-?\d+\.?\d*)\s+",
                    'cm^3/g': r"Average loading excess \[cm\^3 \(STP\)/gr framework\]\s+(-?\d+\.?\d*)\s+",
                    'mol/kg': r"Average loading excess \[mol/kg framework\]\s+(-?\d+\.?\d*)\s+",
                    'mg/g': r"Average loading excess \[milligram/gram framework\]\s+(-?\d+\.?\d*)\s+",
                    'cm^3/cm^3': r"Average loading excess \[cm\^3 \(STP\)/cm\^3 framework\]\s+(-?\d+\.?\d*)\s+"
                    }
        if unit not in patterns:
            raise ValueError(f"单位错误！不支持的单位: {unit}")
        
        result = {}
        # 由于RASPA输出可能包含多个component的加载量，这里需要确保匹配顺序
        # 通过在正则表达式前加上 Component \d+ \[(.*?)\] 来匹配特定component的块
        # 但更简单和鲁棒的方式是直接匹配所有该单位的 loading，然后按顺序分配给 self.components
        data = re.findall(patterns[unit], self.output_string)
        
        if len(data) != len(self.components):
            # 如果匹配到的数据量和组件数量不符，可能需要更复杂的解析或警告
            # 暂时先这样处理，如果遇到问题再细化
            print(f"警告: {unit} 的超额吸附量数据量 ({len(data)}) 与组件数量 ({len(self.components)}) 不符。")

        for i, comp_name in enumerate(self.components):
            if i < len(data):
                result[comp_name] = data[i]
            else:
                result[comp_name] = "N/A" # 如果数据不足，标记为不可用
        return result

    def get_absolute_adsorption(self, unit='cm^3/g'):
        '''
        指定单位，返回绝对吸附量，返回值是一个字典，键是吸附质的名称，值是吸附量;
        若不指定单位，默认为cm^3/g
        unit: 'mol/uc','cm^3/g','mol/kg','mg/g','cm^3/cm^3'
        '''
        patterns = {'mol/uc': r"Average loading absolute \[molecules/unit cell\]\s+(-?\d+\.?\d*)\s+",
                    'cm^3/g': r"Average loading absolute \[cm\^3 \(STP\)/gr framework\]\s+(-?\d+\.?\d*)\s+",
                    'mol/kg': r"Average loading absolute \[mol/kg framework\]\s+(-?\d+\.?\d*)\s+",
                    'mg/g': r"Average loading absolute \[milligram/gram framework\]\s+(-?\d+\.?\d*)\s+",
                    'cm^3/cm^3': r"Average loading absolute \[cm\^3 \(STP\)/cm\^3 framework\]\s+(-?\d+\.?\d*)\s+"
                    }
        if unit not in patterns:
            raise ValueError(f"单位错误！不支持的单位: {unit}")
        
        result = {}
        data = re.findall(patterns[unit], self.output_string)

        if len(data) != len(self.components):
            print(f"警告: {unit} 的绝对吸附量数据量 ({len(data)}) 与组件数量 ({len(self.components)}) 不符。")
            
        for i, comp_name in enumerate(self.components):
            if i < len(data):
                result[comp_name] = data[i]
            else:
                result[comp_name] = "N/A"
        return result

# --- 辅助函数 ---

def get_unit_cell(cif_location, cutoff):
    """
    从CIF文件中读取晶格常数，并根据截断半径计算RASPA模拟所需的unit_cell数目。
    """
    flag = [
        '_cell_length_a', '_cell_length_b', '_cell_length_c',
        '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'
    ]
    params = {}
    with open(cif_location, 'r') as f:
        for line in f:
            for f_name in flag:
                if f_name in line:
                    try:
                        value = float(line.strip().split()[1].split('(')[0]) # 处理 (N) 格式
                        params[f_name] = value
                    except (ValueError, IndexError):
                        pass # 忽略无法解析的行

    # 检查是否所有参数都已找到
    if not all(f in params for f in flag):
        missing = [f for f in flag if f not in params]
        raise ValueError(f"CIF文件 '{cif_location}' 中缺少必要的晶格参数: {missing}")

    a = params['_cell_length_a']
    b = params['_cell_length_b']
    c = params['_cell_length_c']
    alpha = params['_cell_angle_alpha'] * math.pi / 180
    beta = params['_cell_angle_beta'] * math.pi / 180
    gamma = params['_cell_angle_gamma'] * math.pi / 180

    # 计算晶胞体积
    V = a * b * c * (1 + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma) - 
                     (math.cos(alpha))**2 - (math.cos(beta))**2 - (math.cos(gamma))**2) ** 0.5
    
    # 计算晶胞各个面的表面积
    base_area_x = b * c * math.sin(alpha)
    base_area_y = a * c * math.sin(beta)
    base_area_z = a * b * math.sin(gamma)

    # 计算各个方向的最小距离 (高)
    perpendicular_length_x = V / base_area_x
    perpendicular_length_y = V / base_area_y
    perpendicular_length_z = V / base_area_z

    # 根据截断半径r_cut计算所需各方向的unit_cell数目
    # RASPA 的 unit cell 推荐是 2 * cutoff / min_dimension
    a_unitcell = math.ceil(2 * cutoff / perpendicular_length_x)
    b_unitcell = math.ceil(2 * cutoff / perpendicular_length_y)
    c_unitcell = math.ceil(2 * cutoff / perpendicular_length_z)

    # 确保最小为 1，因为单元格数不能为 0
    a_unitcell = max(1, a_unitcell)
    b_unitcell = max(1, b_unitcell)
    c_unitcell = max(1, c_unitcell)

    return "{} {} {}".format(a_unitcell, b_unitcell, c_unitcell)


def generate_simulation_input(template: str, cutoff: float, cif_name: str,
                              unitcell: str, heliumvoidfraction):
    """
    根据模板和给定参数生成RASPA模拟的输入文件内容。
    注意：这个函数现在直接接收 cif_name, unitcell, heliumvoidfraction，
    不再负责从文件中读取这些信息。
    """
    # 检查 heliumvoidfraction 是否为 NaN，如果是，可能需要替换为默认值或抛出错误
    if pd.isna(heliumvoidfraction):
        # 根据 RASPA 的要求，heliumvoidfraction 必须是数字
        # 这里可以选择抛出错误，或者给一个默认值（例如 0.0，但这可能不准确）
        raise ValueError(f"结构 {cif_name} 的 helium_excess_widom 为 NaN，无法生成RASPA输入文件。")
        
    return template.format(cif_name=cif_name, cutoff=cutoff, unitcell=unitcell, 
                           heliumvoidfraction=heliumvoidfraction)


def get_result(output_str: str, components: list, cif_name: str):
    """
    解析RASPA输出字符串，提取吸附结果。
    """
    res = {}
    units = ['mol/uc', 'cm^3/g', 'mol/kg', 'mg/g', 'cm^3/cm^3']
    res["name"] = cif_name
    output = RASPA_Output_Data(output_str)
    res["finished"] = str(output.is_finished()) # 这里的finished是RASPA内部的finished状态

    res["warning"] = ""
    if res["finished"] == 'True':
        warnings = output.get_warnings()
        if warnings:
            res["warning"] = "; ".join(warnings)
    else: # 如果模拟未完成，所有结果都为空
        for unit in units:
            for c in components:
                res[c + "_absolute_" + unit] = " "
                res[c + "_excess_" + unit] = " "
        res["warning"] = "Simulation incomplete or crashed inside RASPA"
        
    # 如果finished为True，再尝试获取数据
    if res["finished"] == 'True':
        for unit in units:
            try:
                absolute_capacity = output.get_absolute_adsorption(unit=unit)
                excess_capacity = output.get_excess_adsorption(unit=unit)
                for c in components:
                    res[c + "_absolute_" + unit] = absolute_capacity.get(c, "N/A")
                    res[c + "_excess_" + unit] = excess_capacity.get(c, "N/A")
            except ValueError as e:
                # 捕获单位错误，但应在调用前确保单位正确
                print(f"错误: 获取 {cif_name} 的 {unit} 吸附量时发生单位错误: {e}")
                for c in components:
                    res[c + "_absolute_" + unit] = "Error"
                    res[c + "_excess_" + unit] = "Error"
            except Exception as e:
                # 其他解析错误
                print(f"错误: 解析 {cif_name} 的 {unit} 吸附量时发生意外错误: {e}")
                for c in components:
                    res[c + "_absolute_" + unit] = "ParseError"
                    res[c + "_excess_" + unit] = "ParseError"

    return res


def get_field_headers(components: list):
    """
    生成结果CSV文件的列标题。
    """
    headers = ["name", "finished"] # finished 字段反映甲烷模拟是否完成
    units = ['mol/uc', 'cm^3/g', 'mol/kg', 'mg/g', 'cm^3/cm^3']
    for i in ["absolute", "excess"]:
        for j in components:
            for unit in units:
                headers.append(j + "_" + i + "_" + unit)
    headers.append("warning")
    return headers


def get_components_from_input(input_text: str):
    """
    从RASPA输入模板中提取吸附质的名称。
    """
    # 查找 MoleculeName，并捕获其后的内容，直到行尾
    components = re.findall(r'MoleculeName\s+(.+)', input_text)
    # 进一步清理可能存在的注释或其他杂项，只保留第一个单词
    cleaned_components = [comp.strip().split()[0] for comp in components]
    return cleaned_components


def write_result(result_file, result: dict, headers: list):
    """
    将模拟结果写入CSV文件。
    """
    with open(result_file, 'a', newline='', encoding='utf-8') as f: # 明确指定编码和 newline
        import csv
        writer = csv.writer(f)
        # 确保每个 header 都能在 result 字典中找到对应的值，如果找不到则为空字符串
        row_values = [str(result.get(header, " ")) for header in headers]
        writer.writerow(row_values)


def write_error(result_file, cif_name, headers: list, error_msg="Simulation Error"):
    """
    将错误信息写入CSV文件，标记为未完成。
    """
    with open(result_file, 'a', newline='', encoding='utf-8') as f: # 明确指定编码和 newline
        import csv
        writer = csv.writer(f)
        error_row_data = []
        for header in headers:
            if header == "name":
                error_row_data.append(cif_name)
            elif header == "finished":
                error_row_data.append("Error") # 标记为错误
            elif header == "warning":
                error_row_data.append(error_msg)
            else:
                error_row_data.append(" ") # 其他数据为空
        writer.writerow(error_row_data)


def check_parameters():
    """
    检查配置文件中的参数，并进行初步验证。
    """
    cur_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(cur_path) # 切换到脚本所在目录，确保相对路径正确
    
    config = configparser.ConfigParser()
    
    config_file_path = "config.ini"
    if not os.path.exists(config_file_path):
        print(f"错误: 配置文件 '{config_file_path}' 未找到。请确保它与脚本在同一目录。")
        exit(1)
    config.read(config_file_path, encoding='utf8')
    
    section = "ADSORPTION_CONFIG"
    # 确保键与 config.ini 严格匹配 (configparser 对键不区分大小写，但为一致性明确指定)
    # timeout_seconds 是一个新增的推荐参数
    full_options = ['RASPA_dir', 'cif_location', 'max_threads', 'CutOffVDM', 'helium_data_file', 'timeout_seconds'] 
    
    options_in_config = [opt.lower() for opt in config.options(section)] # 转换为小写进行匹配
    missing_options = []
    option_dic = {}
    
    for op in full_options:
        # 使用 lower() 获取配置值，但以原始大小写存储在字典中
        if op.lower() not in options_in_config:
            missing_options.append(op)
        else:
            option_dic[op] = config.get(section, op.lower())

    if len(missing_options) > 0:
        print("配置文件中参数不完整! (The parameters in the configuration file are incomplete !)")
        print("缺少的选项 (missing options) : " + str(missing_options))
        exit(1)

    raspa_dir = option_dic['RASPA_dir']
    cif_location_raw = option_dic['cif_location'] # 保持原始值用于文件/目录检查
    cutoffvdm = option_dic['CutOffVDM']
    max_threads = option_dic['max_threads']
    helium_data_file = option_dic['helium_data_file']
    # 从配置中直接获取，提供默认值以防用户未在config.ini中设置
    timeout_seconds = option_dic.get('timeout_seconds', 3600) 

    # 路径转换为绝对路径
    raspa_dir = os.path.abspath(raspa_dir)
    cif_location_raw = os.path.abspath(cif_location_raw)
    helium_data_file = os.path.abspath(helium_data_file)

    # 路径存在性检查
    simulate_bin_path = os.path.join(raspa_dir, "bin", "simulate")
    if not os.path.exists(simulate_bin_path) or not os.path.isfile(simulate_bin_path):
        print(f'错误: RASPA模拟器文件无效或不存在！请检查RASPA_dir配置: {simulate_bin_path}')
        exit(1)

    if not os.path.exists(cif_location_raw):
        print(f'错误: cif目录或文件无效！(Invalid cif_location!): {cif_location_raw}')
        exit(1)
    
    if not os.path.exists(helium_data_file):
        print(f"错误: 氦气数据文件 '{helium_data_file}' 不存在，请检查配置。")
        exit(1)

    # 类型转换和数值范围检查
    try:
        cutoffvdm = float(cutoffvdm)
        if not (0 < cutoffvdm < 100): # 简单的合理性检查
             print("警告: 截断半径CutOffVDM的值可能不常见。")
    except ValueError:
        print("错误: 截断半径必须为数字！(CutOffVDM must be numerical !)")
        exit(1)

    try:
        max_threads = int(max_threads)
        if max_threads <= 0:
            print("错误: 线程数必须为正整数！(max_threads must be a positive integer !)")
            exit(1)
    except ValueError:
        print("错误: 线程数必须为整数！(max_threads must be integer !)")
        exit(1)
    
    try:
        timeout_seconds = int(timeout_seconds)
        if timeout_seconds <= 0:
            print("错误: 超时时间必须为正整数秒！(timeout_seconds must be a positive integer seconds !)")
            exit(1)
    except ValueError:
        print("错误: 超时时间必须为整数！(timeout_seconds must be integer !)")
        exit(1)

    # 获取CIF文件列表
    cifs = []
    cif_dir = ""
    if os.path.isfile(cif_location_raw): # 如果 cif_location 是一个文件
        cifs.append(os.path.basename(cif_location_raw))
        cif_dir = os.path.dirname(cif_location_raw)
    else: # 如果 cif_location 是一个目录
        cif_dir = cif_location_raw
        all_files = os.listdir(cif_dir)
        for cif in all_files:
            if cif.endswith('.cif'):
                cifs.append(cif)
    
    if len(cifs) == 0:
        print('错误: cif目录中缺乏有效的cif文件！(There are no valid cif files in the cif_location)')
        exit(1)

    return raspa_dir, cif_dir, cifs, cutoffvdm, max_threads, helium_data_file, timeout_seconds


def work(cif_dir: str, cif_file: str, RASPA_dir: str, result_file: str, 
         components: list, headers: list, input_text: str, lock: Lock, q: Queue, 
         timeout_seconds: int = 1800):
    """
    执行RASPA模拟，并处理结果写入。
    使用 subprocess.run 替代 os.system，避免 os.chdir。
    """
    cif_name = cif_file[:-4] # 移除 .cif 后缀
    
    # 构建当前结构对应的输出目录
    cur_path = os.path.abspath(os.path.dirname(__file__))
    output_root_dir = os.path.join(cur_path, "RASPA_Output") # 根输出目录
    cmd_dir = os.path.join(output_root_dir, cif_name) # 当前结构专用目录
    
    # 确保当前结构的输出目录存在
    os.makedirs(cmd_dir, exist_ok=True) 

    # 复制CIF文件到当前结构的输出目录
    shutil.copy(os.path.join(cif_dir, cif_file), cmd_dir)
    
    # 写入 simulation.input 文件
    input_file_path = os.path.join(cmd_dir, "simulation.input")
    with open(input_file_path, "w") as f1:
        f1.write(input_text)

    # 构建RASPA执行命令
    raspa_exe_path = os.path.join(RASPA_dir, "bin", "simulate")
    cmd = [raspa_exe_path, "simulation.input"] # 使用列表形式更安全

    print(f"Info: 启动 {cif_name} 的 RASPA 模拟...")
    try:
        # 执行RASPA命令
        result_process = subprocess.run(
            cmd,
            cwd=cmd_dir, # 在指定目录下执行命令
            capture_output=True, # 捕获标准输出和标准错误
            text=True, # 以文本模式捕获输出
            check=False, # 不在非零退出码时抛出异常，我们手动检查 returncode
            timeout=timeout_seconds # 设置超时时间
        )
        
        # 检查RASPA进程的退出码
        if result_process.returncode == 0:
            print(f"Info: RASPA for {cif_name} 进程正常结束 (return code 0)。")
            raspa_output_system_dir = os.path.join(cmd_dir, "Output", "System_0")
            
            output_data_files = []
            if os.path.exists(raspa_output_system_dir):
                # 查找所有 .data 文件
                output_data_files = [f for f in os.listdir(raspa_output_system_dir) if f.endswith('.data')]
            
            if output_data_files:
                # 假设找到的第一个 .data 文件是正确的输出
                output_file_name = output_data_files[0]
                full_output_file_path = os.path.join(raspa_output_system_dir, output_file_name)
                
                with open(full_output_file_path, 'r') as f2:
                    output_str = f2.read()
                
                # 解析RASPA输出获取结果
                result_data = get_result(output_str, components, cif_name)
                
                # 写入最终结果文件（加锁以确保线程安全）
                lock.acquire()
                try:
                    write_result(result_file, result_data, headers)
                    print(f"\033[0;30;42m\n{cif_name} 模拟已完成并成功写入结果。\n\033[0m")
                finally:
                    lock.release()
            else:
                # 即使RASPA进程返回0，如果未找到.data文件，也视为失败
                error_message = (f"RASPA for {cif_name} 进程正常退出，但在 '{raspa_output_system_dir}' 中未找到 '.data' 输出文件。\n"
                                 f"RASPA 标准输出:\n{result_process.stdout}\n"
                                 f"RASPA 标准错误:\n{result_process.stderr}")
                print(f"\033[0;37;41m\n{error_message}\n\033[0m")
                lock.acquire()
                try:
                    write_error(result_file, cif_name, headers, error_message)
                finally:
                    lock.release()
        else:
            # RASPA 进程以非零退出码结束，通常表示模拟失败
            error_message = (f"RASPA 进程 {cif_name} 以非零退出码 {result_process.returncode} 退出。\n"
                             f"标准输出:\n{result_process.stdout}\n"
                             f"标准错误:\n{result_process.stderr}")
            print(f"\033[0;37;41m\n{error_message}\n\033[0m")
            lock.acquire()
            try:
                write_error(result_file, cif_name, headers, error_message)
            finally:
                lock.release()

    except subprocess.TimeoutExpired:
        # 模拟超时
        error_message = f"RASPA 模拟 {cif_name} 在 {timeout_seconds} 秒后超时。"
        print(f"\033[0;37;41m\n{error_message}\n\033[0m")
        lock.acquire()
        try:
            write_error(result_file, cif_name, headers, error_message)
        finally:
            lock.release()
    except FileNotFoundError:
        # 找不到 RASPA 可执行文件
        error_message = f"RASPA 可执行文件或必要文件未找到。请检查RASPA_dir和路径: {raspa_exe_path}"
        print(f"\033[0;37;41m\n{error_message}\n\033[0m")
        lock.acquire()
        try:
            write_error(result_file, cif_name, headers, error_message)
        finally:
            lock.release()
    except Exception as e:
        # 其他所有未预期的错误
        error_message = f"处理 {cif_name} 时发生意外错误: {repr(e)}"
        print(f"\033[0;37;41m\n{error_message}\n\033[0m")
        lock.acquire()
        try:
            write_error(result_file, cif_name, headers, error_message)
        finally:
            lock.release()
    finally:
        q.put(1) # 无论成功或失败，都将令牌放回队列，以便其他任务可以启动


# --- 主逻辑函数 ---
def main():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(cur_path) # 确保脚本在自己的目录下运行，便于管理文件

    # 1. 检查并加载配置参数
    raspa_dir, cif_dir, cifs, cutoffvdm, max_threads, helium_data_file, timeout_seconds = check_parameters()
    
    lock = Lock() # 用于线程写入结果文件的锁

    # 2. 加载氦气探测结果文件 (用于获取 helium_excess_widom)
    print(f"Info: 正在加载氦气数据文件 '{helium_data_file}'...")
    helium_df = None
    try:
        if helium_data_file.lower().endswith('.csv'):
            helium_df = pd.read_csv(helium_data_file, index_col=False) 
            print(f"Info: CSV文件 '{helium_data_file}' 已加载为DataFrame。")
        elif helium_data_file.lower().endswith(('.xls', '.xlsx')):
            helium_df = pd.read_excel(helium_data_file)
            print(f"Info: XLSX文件 '{helium_data_file}' 已加载为DataFrame。")
        else:
            raise ValueError(f"不支持的氦气数据文件格式: {os.path.basename(helium_data_file)}。请使用 .csv, .xls 或 .xlsx 文件。")

        if 'name' in helium_df.columns:
            helium_df = helium_df.set_index('name')
            print(f"Info: 'name' 列已设置为氦气数据 DataFrame 索引。")
        else:
            raise ValueError(f"氦气数据文件 '{helium_data_file}' 中未找到 'name' 列。无法进行后续处理。")

        # 将氦气结果中的 'finished' 列重命名为 'finished_helium' 并转换为布尔型
        helium_df['finished_helium'] = helium_df['finished'].replace('Error', False).fillna(False).astype(bool)
        if 'finished' in helium_df.columns: # 确保删除原始列
            helium_df = helium_df.drop(columns=['finished'])
        
        print(f"Info: 氦气数据文件 '{helium_data_file}' 加载完成并处理。")
    except Exception as e:
        print(f"错误: 无法加载或解析氦气数据文件 '{helium_data_file}': {e}")
        exit(1)

    # 3. 加载 RASPA 模拟模板文件
    template_file_path = os.path.join(cur_path, "simulation_template.input")
    if not os.path.exists(template_file_path):
        print(f"错误: 模拟模板文件 '{template_file_path}' 未找到。")
        exit(1)
    
    with open(template_file_path, "r") as f:
        template = f.read()
    print(f"Info: 模拟模板文件 '{template_file_path}' 加载完成。")

    # 从模板中获取吸附质组件名称 (例如：甲烷通常是 'Methane')
    components = get_components_from_input(template)
    if not components:
        print("警告: 未能在 simulation_template.input 中找到任何 'MoleculeName'。请检查模板文件。")
        print("警告: 无法确定吸附质名称，结果文件中的吸附量列可能不完整。")
        # 建议此处退出，因为不知道吸附质会导致结果解析错误
        exit(1) 
    
    headers = get_field_headers(components) # 生成最终结果CSV的表头

    # 4. 初始化甲烷吸附结果文件 (adsorption_results.csv)
    result_file = os.path.join(cur_path, "adsorption_results.csv")
    
    # 尝试加载旧的甲烷吸附结果文件，以实现断点续算功能
    methane_results_df = None
    if os.path.exists(result_file):
        try:
            # 读取时注意可能存在“Error”字符串
            methane_results_df = pd.read_csv(result_file, index_col='name', dtype={'finished': str})
            # 处理 'finished' 列，这里是甲烷模拟的 finished 状态
            methane_results_df['finished_methane'] = methane_results_df['finished'].replace('Error', False).fillna(False).astype(bool)
            print(f"Info: 发现旧的甲烷吸附结果文件 '{result_file}'，将基于其状态继续模拟。")
            
            # 如果旧文件存在且有效，将其内容重新写入（或构建一个要写入的字典），以保持文件的连续性
            # 并避免在多线程环境下重复写入已完成的任务
            # 更好的做法是在循环前，将所有已完成的旧结果复制到新结果文件，然后只处理未完成的
            # 为简化，我们直接在循环中根据状态跳过
            with open(result_file, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(headers) # 写入新的表头
                # 重新写入已完成的旧数据
                for idx, row in methane_results_df.iterrows():
                    if row['finished_methane']: # 如果甲烷模拟已完成
                        row_dict = row.to_dict()
                        row_dict['name'] = idx # index_col 变为 name
                        row_dict['finished'] = 'True' if row['finished_methane'] else 'Error' # 恢复为字符串
                        write_result(f, row_dict, headers) # 写入这一行
            print(f"Info: 旧的甲烷吸附结果文件中已完成的任务已重新写入 '{result_file}'。")

        except Exception as e:
            print(f"警告: 无法读取旧的甲烷吸附结果文件 '{result_file}' ({e})，将重新创建。")
            methane_results_df = None # 读取失败则视为没有旧文件
    
    if methane_results_df is None or not os.path.exists(result_file): # 再次检查，确保文件已初始化
        if os.path.exists(result_file): # 避免重复删除
            os.remove(result_file)
        with open(result_file, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Info: 已创建新的甲烷结果文件 '{result_file}' 并写入表头。")


    # 5. 初始化 RASPA 输出目录
    output_root_dir = os.path.join(cur_path, "RASPA_Output")
    if os.path.exists(output_root_dir):
        # 如果目录存在，给用户一个明确的警告并退出，避免覆盖或混淆数据
        print(f"警告: RASPA_Output目录已存在 ('{output_root_dir}')。为避免数据混合，请手动删除或备份后再运行。")
        exit(1) 
    os.makedirs(output_root_dir)
    print(f"Info: 已创建输出根目录 '{output_root_dir}'。")

    # 6. 设置线程池 (使用 Queue 控制并发)
    q = Queue(maxsize=max_threads)
    for i in range(max_threads):
        q.put(1) # 放入令牌，允许 max_threads 个线程同时运行

    active_threads = [] # 存储所有已启动的线程对象

    # 7. 遍历 CIF 文件并启动模拟任务
    for cif_file_name in cifs: # 这里的 cifs 是 cif 文件名的列表，例如 ['HKUST-1.cif', 'MOF-5.cif']
        q.get() # 尝试从队列中获取一个令牌，如果队列为空则阻塞，直到有令牌可用

        cif_name = cif_file_name[:-4] # 获取结构名称，例如 'HKUST-1'
        
        # 7.1. 检查甲烷模拟是否已完成 (实现断点续算)
        if methane_results_df is not None and cif_name in methane_results_df.index:
            if methane_results_df.loc[cif_name]['finished_methane'] == True:
                print(f"Info: '{cif_name}' 的甲烷吸附模拟已在结果文件 (adsorption_results.csv) 中标记为完成，跳过此模拟。")
                q.put(1) # 释放队列令牌
                continue
            elif methane_results_df.loc[cif_name]['finished_methane'] == False:
                 print(f"Info: '{cif_name}' 的甲烷吸附模拟之前失败，将尝试重新运行。")
        
        # 7.2. 从氦气结果文件中获取 helium_excess_widom
        cif_data_helium = None
        try:
            cif_data_helium = helium_df.loc[cif_name] # 这里的 helium_df 是来自 helium 结果
        except KeyError:
            error_msg = f"氦气数据文件 '{helium_data_file}' 中未找到 '{cif_name}' 的相关数据。"
            print(f"警告: {error_msg} 跳过甲烷模拟。")
            lock.acquire()
            try:
                write_error(result_file, cif_name, headers, error_msg)
            finally:
                lock.release()
            q.put(1) # 释放队列令牌
            continue

        # 7.3. 检查 helium_excess_widom 是否可用且氦气模拟已完成
        if 'helium_excess_widom' not in cif_data_helium.index or \
           pd.isna(cif_data_helium['helium_excess_widom']) or \
           not cif_data_helium['finished_helium']: # 必须确认氦气模拟已完成
            
            error_msg = f"'{cif_name}' 的 'helium_excess_widom' 缺失、为 NaN 或氦气模拟未完成。"
            print(f"警告: {error_msg} 跳过甲烷模拟。")
            lock.acquire()
            try:
                write_error(result_file, cif_name, headers, error_msg)
            finally:
                lock.release()
            q.put(1) # 释放队列令牌
            continue
        
        # 获取 helium_excess_widom 值
        heliumvoidfraction = cif_data_helium['helium_excess_widom']

        print(f"Info: 正在为 '{cif_name}' 准备甲烷吸附模拟 (HeliumVoidFraction: {heliumvoidfraction:.6f})...")

        # 7.4. 计算 unit_cell 数目
        unitcell_str = None
        try:
            unitcell_str = get_unit_cell(os.path.join(cif_dir, cif_file_name), cutoffvdm)
        except Exception as e:
            error_msg = f"计算 '{cif_name}' 的 unit_cell 失败: {e}"
            print(f"错误: {error_msg}")
            lock.acquire()
            try:
                write_error(result_file, cif_name, headers, error_msg)
            finally:
                lock.release()
            q.put(1) # 释放队列令牌
            continue

        # 7.5. 生成 RASPA simulation.input 内容
        input_text_for_simulation = None
        try:
            input_text_for_simulation = generate_simulation_input(
                template=template, 
                cutoff=cutoffvdm, 
                cif_name=cif_name, 
                unitcell=unitcell_str, 
                heliumvoidfraction=heliumvoidfraction
            )
        except ValueError as e:
            error_msg = f"生成 '{cif_name}' 的 simulation.input 失败: {e}"
            print(f"错误: {error_msg}")
            lock.acquire()
            try:
                write_error(result_file, cif_name, headers, error_msg)
            finally:
                lock.release()
            q.put(1) # 释放队列令牌
            continue


        # 7.6. 启动 RASPA 模拟线程
        thread = threading.Thread(
            target=work, 
            args=(cif_dir, cif_file_name, raspa_dir, result_file, components, headers, 
                  input_text_for_simulation, lock, q, timeout_seconds),
            name=f"RASPA_{cif_name}_Thread" # 给线程一个有意义的名字
        )
        thread.start()
        active_threads.append(thread) # 将线程添加到列表中，以便后续等待其完成

    # 8. 等待所有线程完成
    for t in active_threads:
        t.join() # 阻塞主线程，直到所有子线程执行完毕

    print("\033[0;30;42m\n所有甲烷吸附模拟任务完成！(All Methane Adsorption tasks finished)\n\033[0m")


if __name__ == '__main__':
    main()
