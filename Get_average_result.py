import os
import re
import numpy as np

# define the directory that stores results
output_dir = 'result/Urban_cos'

# containers for acc, nmi, kappa, ca, and runtime values
acc_list = []
nmi_list = []
kappa_list = []
ca_epoch_list = []
elapsed_time_list = []

# regular expression to parse acc, nmi, kappa, ca, and elapsed time
pattern = re.compile(r"clustering_result:.*?acc:\s*([0-9\.]+).*?nmi:\s*([0-9\.]+).*?kappa:\s*([0-9\.]+).*?ca\s*\[([0-9\.\s]+)\]\s*Elapsed time:\s*([0-9\.]+)")
if __name__ == '__main__':
    # iterate over every file inside the directory
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.txt'):  # only process .txt files
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, 'r') as f:
                content = f.read()

                # look for the line that contains clustering metrics
                match = pattern.search(content)

                if match:
                    try:
                        acc_list.append(float(match.group(1)))
                        nmi_list.append(float(match.group(2)))
                        kappa_list.append(float(match.group(3)))
                        ca_values = [float(x) for x in match.group(4).split() if x.strip()]
                        if len(ca_epoch_list) == 0:
                            ca_epoch_list = np.array(ca_values)
                        else:
                            ca_epoch_list += np.array(ca_values)
                        elapsed_time_list.append(float(match.group(5)))
                    except ValueError:
                        print(f"Error parsing values in file: {file_name}")
                else:
                    print(f"No match found in file: {file_name}")

    # compute averages while guarding against empty lists
    num_epochs = len(acc_list)
    average_acc = np.mean(acc_list) if acc_list else 'N/A'
    average_nmi = np.mean(nmi_list) if nmi_list else 'N/A'
    average_kappa = np.mean(kappa_list) if kappa_list else 'N/A'
    average_ca = (ca_epoch_list / num_epochs) if num_epochs > 0 else 'N/A'
    average_elapsed_time = np.mean(elapsed_time_list) if elapsed_time_list else 'N/A'

    # write aggregated results to final.txt
    final_file = os.path.join(output_dir, 'final.txt')
    with open(final_file, 'w') as f:
        f.write(f"Average acc: {average_acc}\n")
        f.write(f"Average nmi: {average_nmi}\n")
        f.write(f"Average kappa: {average_kappa}\n")
        f.write(f"Average ca: {average_ca}\n")
        f.write(f"Average elapsed time: {average_elapsed_time}\n")

    print(f"Final results written to {final_file}")
