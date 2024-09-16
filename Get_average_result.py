import os
import re
import numpy as np

# 定义结果文件夹
output_dir = 'result/houston0002'

# 存储acc, nmi和kappa的列表
acc_list = []
nmi_list = []
kappa_list = []

# 定义正则表达式匹配clustering_result的行并提取acc, nmi, kappa
pattern = re.compile(r"clustering_result:.*acc:\s*([0-9.]+).*nmi:\s*([0-9.]+).*kappa:\s*([0-9.]+)")
if __name__ == '__main__':
    # 遍历指定目录下的所有文件
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.txt'):  # 只处理 .txt 文件
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, 'r') as f:
                lines = f.readlines()

                # 查找包含clustering_result的行
                for line in lines:
                    match = pattern.search(line)

                    if match:
                        acc_list.append(float(match.group(1)))
                        nmi_list.append(float(match.group(2)))
                        kappa_list.append(float(match.group(3)))
                        break

    # 计算平均值和方差
    average_acc = np.mean(acc_list)
    average_nmi = np.mean(nmi_list)
    average_kappa = np.mean(kappa_list)

    variance_acc = np.var(acc_list)
    variance_nmi = np.var(nmi_list)
    variance_kappa = np.var(kappa_list)

    # 将结果写入final.txt文件
    final_file = os.path.join(output_dir, 'final.txt')
    with open(final_file, 'w') as f:
        f.write(f"Average acc: {average_acc}\n")
        f.write(f"Variance acc: {variance_acc}\n")
        f.write(f"Average nmi: {average_nmi}\n")
        f.write(f"Variance nmi: {variance_nmi}\n")
        f.write(f"Average kappa: {average_kappa}\n")
        f.write(f"Variance kappa: {variance_kappa}\n")

    print(f"Final results written to {final_file}")
