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

class RASPA_Output_Data():
    '''
        RASPA输出文件对象
    '''
    '''
    示例：
        with open('./output.data','r') as f:
            str = f.read()
        output = RASPA_Output_Data(str)
        print(output.is_finished())
        print(output.get_absolute_adsorption())

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
        return result[0]

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
        if unit not in patterns.keys():
            raise ValueError('单位错误！')
        result = {}
        data = re.findall(patterns[unit], self.output_string)
        for i, j in zip(self.components, data):
            result[i] = j
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
        if unit not in patterns.keys():
            raise ValueError('单位错误！')
        result = {}
        data = re.findall(patterns[unit], self.output_string)
        for i, j in zip(self.components, data):
            result[i] = j
        return result




def get_unit_cell(cif_location, cutoff):

    # 从cif文件中读取晶格常数
    flag = [
        '_cell_length_a', '_cell_length_b', '_cell_length_c',
        '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'
    ]
    flag_length = len(flag) 
    with open(cif_location, 'r') as f:
        i = 1
        flag_count = 0
        while i <= 5e4:
            flag_content = flag[flag_count]
            line = f.readline()
            if flag_content in line:
                if flag_content == '_cell_length_a':
                    a = float(line.strip().split()[1])
                elif flag_content == '_cell_length_b':
                    b = float(line.strip().split()[1])
                elif flag_content == '_cell_length_c':
                    c = float(line.strip().split()[1])
                elif flag_content == '_cell_angle_alpha':
                    alpha = float(line.strip().split()[1]) * math.pi / 180 # 计算时用弧度单位    
                elif flag_content == '_cell_angle_beta':
                    beta = float(line.strip().split()[1]) * math.pi / 180   # 计算时用弧度单位
                elif flag_content == '_cell_angle_gamma':
                    gamma = float(line.strip().split()[1]) * math.pi / 180 # 计算时用弧度单位
                else:
                    print(f'No specific operation for flag_content: {flag_content}!')
                
                flag_count += 1
                if flag_count == flag_length:
                    break
            i += 1

    # 计算晶胞体积          
    V =  a * b * c * (1 + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma) - (math.cos(alpha))**2 - (math.cos(beta))**2 - (math.cos(gamma))**2) ** 0.5
    
    # 计算晶胞各个面的表面积
    base_area_x = b * c * math.sin(alpha)
    base_area_y = a * c * math.sin(beta)
    base_area_z = a * b * math.sin(gamma)

    # 计算各个方向的最小距离，即平行六面体各个方向的高，等于体积除以底面积
    perpendicular_length_x = V / base_area_x
    perpendicular_length_y = V / base_area_y
    perpendicular_length_z = V / base_area_z

    # 根据截断半径r_cut计算所需各方向的unit_cell数目
    a_unitcell = math.ceil(2 * cutoff / perpendicular_length_x)
    b_unitcell= math.ceil(2 * cutoff / perpendicular_length_y)
    c_unitcell= math.ceil(2 * cutoff / perpendicular_length_z)

    return "{} {} {}".format(a_unitcell, b_unitcell, c_unitcell)


def generate_simulation_input(template: str, cutoff: float, cif_dir: str,
                              cif_file: str, file_path):
    unitcell = get_unit_cell(os.path.join(cif_dir, cif_file), cutoff)
    cif_name = cif_file[:-4]
    df = pd.read_csv(file_path)

    # 过滤数据
    filtered_data = df[df['name'] == cif_name]

    # 检查过滤后的数据是否为空
    if filtered_data.empty:
        print(f"警告: 未找到 {cif_name} 的相关数据，跳过模拟。")
        return None  # 或根据需要抛出异常
    
    # 检查 'helium_excess_widom' 列是否缺失或为 NaN
    if 'helium_excess_widom' not in filtered_data.columns or pd.isna(filtered_data['helium_excess_widom'].values[0]):
        print(f"警告: {cif_name} 的 'helium_excess_widom' 缺失或为 NaN，跳过模拟。")
        return None  # 或根据需要抛出异常

    # 获取 helium void fraction 的值
    heliumvoidfraction = filtered_data['helium_excess_widom'].values[0]
    
    return template.format(cif_name=cif_name, cutoff=cutoff, unitcell=unitcell, heliumvoidfraction=heliumvoidfraction)


def work(cif_dir: str, cif_file: str, RASPA_dir: str, result_file: str, components: str, headers: str, input_text: str,
         lock: Lock, q: Queue):
    cif_name = cif_file[:-4]
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(curr_dir, "RASPA_Output")
    cmd_dir = os.path.join(output_dir, cif_name)
    if not os.path.exists(cmd_dir):
        os.makedirs(cmd_dir)
    shutil.copy(os.path.join(cif_dir, cif_file), cmd_dir)
    cmd = os.path.join(RASPA_dir, "bin", "simulate") + " simulation.input"
    with open(os.path.join(cmd_dir, "simulation.input"), "w") as f1:
        f1.write(input_text)
        f1.close()
    os.chdir(cmd_dir)
    if os.system(cmd) == 0:
        lock.acquire()
        try:
            output_file = os.listdir(os.path.join(
                cmd_dir, "Output", "System_0"))[0]
            with open(os.path.join(cmd_dir, "Output", "System_0", output_file), 'r') as f2:
                result = get_result(f2.read(), components, cif_name)
                f2.close()
            write_result(result_file, result, headers)
            print("\033[0;30;42m\n{} has completed\n\033[0m".format(
                cif_name))
        except Exception as e:
            write_error(result_file, cif_name)
            print("\033[0;37;41m\n{} error: {} !\n\033[0m".format(
                cif_name, repr(e)))
        lock.release()
    else:
        lock.acquire()
        write_error(result_file, cif_name)
        print("\033[0;37;41m\n{} error !\n\033[0m".format(
            cif_name))
        lock.release()
    q.put(1)


def get_result(output_str: str, components: list, cif_name: str):
    res = {}
    units = ['mol/uc', 'cm^3/g', 'mol/kg', 'mg/g', 'cm^3/cm^3']
    res["name"] = cif_name
    output = RASPA_Output_Data(output_str)
    res["finished"] = str(output.is_finished())
    res["warning"] = ""
    if res["finished"] == 'True':
        for w in output.get_warnings():
            res["warning"] += (w + "; ")

        for unit in units:
            absolute_capacity = output.get_absolute_adsorption(unit=unit)
            excess_capacity = output.get_excess_adsorption(unit=unit)
            for c in components:
                res[c + "_absolute_" + unit] = absolute_capacity[c]
                res[c + "_excess_" + unit] = excess_capacity[c]
    else:
        for unit in units:
            for c in components:
                res[c + "_absolute_" + unit] = " "
                res[c + "_excess_" + unit] = " "
    return res


def get_field_headers(components: list):
    headers = ["name", "finished"]
    units = ['mol/uc', 'cm^3/g', 'mol/kg', 'mg/g', 'cm^3/cm^3']
    for i in ["absolute", "excess"]:
        for j in components:
            for unit in units:
                headers.append(j + "_" + i + "_" + unit)
    headers.append("warning")
    return headers


def get_components_from_input(input_text: str):
    components = re.findall(r'MoleculeName\s+(.+)', input_text)
    return components


def write_result(result_file, result: dict, headers: list):
    with open(result_file, 'a') as f:
        for i in range(len(headers)):
            if i != len(headers) - 1:
                f.write(result[headers[i]] + ",")
            else:
                f.write(result[headers[i]] + "\n")
        f.close()


def write_error(result_file, cif_name):
    with open(result_file, 'a') as f:
        f.write(cif_name + ",Error,\n")
        f.close()


def check_parameters():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(cur_path)
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf8')
    section = "ADSORPTION_CONFIG"
    full_options = ['raspa_dir', 'cif_location', 'cutoffvdm', 'max_threads','file_path']
    options_in_config = config.options(section)
    missing_options = []
    option_dic = {}
    for op in full_options:
        if op not in options_in_config:
            missing_options.append(op)
        else:
            option_dic[op] = config.get(section, op)

    if len(missing_options) > 0:
        print("配置文件中参数不完整! (The parameters in the configuration file are incomplete !)")
        print("缺少的选项 (missing options) : " + str(missing_options))
        exit()

    raspa_dir = option_dic['raspa_dir']
    cif_dir = option_dic['cif_location']
    cutoffvdm = option_dic['cutoffvdm']
    max_threads = option_dic['max_threads']
    file_path=option_dic['file_path']
    

    if len(raspa_dir) > 0:
        raspa_dir = os.path.abspath(raspa_dir)

    if len(cif_dir) > 0:
        cif_dir = os.path.abspath(cif_dir)

    if not os.path.exists(os.path.join(raspa_dir, "bin", "simulate")):
        print('RASPA目录无效！(Invalid RASPA_dir!)')
        exit()

    if not os.path.exists(cif_dir):
        print('cif目录无效！(Invalid cif_location!)')
        exit()

    try:
        cutoffvdm = float(cutoffvdm)
    except:
        print("截断半径必须为数字！(CutOffVDM must be numerical !)")
        exit()

    try:
        max_threads = int(max_threads)
    except:
        print("线程数必须为整数！(max_threads must be integer !)")
        exit()

    if os.path.isfile(cif_dir):
        cifs = []
        cifs.append(os.path.basename(cif_dir))
        cif_dir = os.path.dirname(cif_dir)
        return raspa_dir, cif_dir, cifs, cutoffvdm, max_threads

    cifs = os.listdir(cif_dir)
    dels = []
    for cif in cifs:
        if not cif.endswith('.cif'):
            dels.append(cif)
    for s in dels:
        cifs.remove(s)
    if len(cifs) == 0:
        print('cif目录中缺乏有效的cif文件！(There are no valid cif files in the cif_location)')
        exit()

    return raspa_dir, cif_dir, cifs, cutoffvdm, max_threads,file_path


def main():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(cur_path)
    raspa_dir, cif_dir, cifs, cutoffvdm, max_threads,file_path = check_parameters()
    lock = Lock()

    with open("./simulation_template.input", "r") as f:
        template = f.read()
    result_file = os.path.join(cur_path, "adsorption_results.csv")
    components = get_components_from_input(template)
    headers = get_field_headers(components)

    if os.path.exists(result_file):
        os.remove(result_file)

    with open(result_file, 'w') as f:
        for i in range(len(headers)):
            if i != len(headers) - 1:
                f.write(headers[i] + ",")
            else:
                f.write(headers[i] + "\n")
        f.close()

    output_dir = os.path.join(cur_path, "RASPA_Output")
    if os.path.exists(output_dir):
        print("RASPA_Output目录已存在，请手动删除后重试！(The RASPA_Output fold already exists, please delete it and try again !)")
        exit()
    os.makedirs(output_dir)

    q = Queue(maxsize=max_threads)
    for i in range(max_threads):
        q.put(1)

    for cif in cifs:
        q.get()
        input_text = generate_simulation_input(template=template, cutoff=cutoffvdm, cif_dir=cif_dir, cif_file=cif, file_path=file_path)
        cif_name = cif[:-4]
        df = pd.read_csv(file_path)
        df['finished'] = df['finished'].replace('Error', False).fillna(False).astype(bool)
        filtered_data = df[df['name'] == cif_name]

    # 检查过滤后的数据是否为空
        if filtered_data.empty:
            print(f"警告: 未找到 {cif_name} 的相关数据，跳过模拟。")
            continue  # 跳过当前的迭代

        heliumvoidfraction = filtered_data['helium_excess_widom'].values[0] if 'helium_excess_widom' in filtered_data.columns else None
        if heliumvoidfraction is None:
            print(f"警告: {cif_name} 的 'helium_excess_widom' 列为空，跳过模拟。")
            continue


    # 确保 'finished' 列存在并且不是 NaN
        if 'finished' in filtered_data.columns and pd.notna(filtered_data['finished'].values[0]):
            finished = filtered_data['finished'].values[0]
        else:
            print(f"警告: {cif_name} 的 'finished' 列为空或无效，跳过模拟。")
            continue  # 跳过当前循环

    # 确保 'helium_excess_widom' 列存在并且不是 NaN
        if 'helium_excess_widom' in filtered_data.columns and pd.notna(filtered_data['helium_excess_widom'].values[0]):
            heliumvoidfraction = filtered_data['helium_excess_widom'].values[0]
        else:
            print(f"警告: {cif_name} 的 'helium_excess_widom' 列为空或无效，跳过模拟。")
            continue  # 跳过当前循环

    # 如果符合条件，继续处理
        print(f"Processing {cif_name} with HeliumVoidFraction {heliumvoidfraction}")

        thread = threading.Thread(target=work, args=(cif_dir, cif, raspa_dir,result_file, components, headers, input_text, lock, q))
        thread.start()
        time.sleep(0.3)
        os.chdir(cur_path)


    for t in threading.enumerate():
        if t.is_alive() and t.getName() != "MainThread":
            t.join()

    print("\033[0;30;42m\n完成！(Finish)\n\033[0m")



if __name__ == '__main__':
    main()
