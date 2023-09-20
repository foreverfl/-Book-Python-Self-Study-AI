import os
import pandas as pd
import matplotlib.pyplot as plt


current_script_path = os.path.abspath(__file__)  # 현재 스크립트 파일의 절대 경로
current_directory = os.path.dirname(current_script_path)
fish_csv_path = os.path.join(current_directory, 'fish.csv')

data = pd.read_csv(fish_csv_path)
print(data)

# Bream과 Smelt 데이터만 추출
bream_data = data[data['Species'] == 'Bream']
smelt_data = data[data['Species'] == 'Smelt']

plt.scatter(bream_data['Length1'], bream_data['Width'],
            label='Bream', color='blue')  # Bream 산점도
plt.scatter(smelt_data['Length1'], smelt_data['Width'],
            label='Smelt', color='red')  # Smelt 산점도
plt.xlabel('Length1')
plt.ylabel('Width')
plt.legend()
plt.show()
