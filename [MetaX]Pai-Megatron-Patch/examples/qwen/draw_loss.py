import re
import pandas   as pd
import matplotlib.pyplot as plt

log_file_path = "llama2-7b_2048_log_2024072510.log"
step_lines = []


with open(log_file_path, 'r') as file:
    for line in file:
        match = re.search(r'iteration(\s+)(\d+)/.*lm loss:\s+(\d+\.*\d*[eE][-+]?\d+)', line)
        if match:
            iter_num = int(match.group(2))
            step_loss = float(match.group(3))
            step_lines.append({"iter": iter_num, "step_loss": step_loss})


df = pd.DataFrame(step_lines)


grouped_df = df.groupby('iter').mean()


plt.plot(grouped_df.index, grouped_df['step_loss'], marker='o')
plt.xlabel('Iteration')
plt.ylabel('Step Loss')
plt.title('Iteration vs  Loss')
plt.grid(True)
plt.savefig('loss.png')