import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 시뮬레이션 설정
time_step = 1  # 시간 간격 (초)
target_temperature = 200  # 목표 온도 (도 섭씨)
initial_temperature = 200  # 초기 온도 (도 섭씨)
conveyor_length = 1000  # 컨베이어 벨트 총 길이
natural_temperature_k = 0.001
natural_temperature = 25
seed = 42
np.random.seed(seed) 
speed = np.random.normal(loc = 2, scale=0.5, size = conveyor_length)  


temperature_tolerance = 10
heating_zones = [10, 100, 200]  # 가열 장치 위치들
# heating_zones = []
heating_power_coef = 10  # 가열 장치의 가열 파워 계수
heating_power_std = heating_power_coef/10  # 가열 장치의 가열 파워 분포의 표준 편차


# 초기화
actual_temperatures = [initial_temperature]
product_positions = [0]
times = [0]

# 가열 설비 조정 함수
def adjust_heating_setting(current_temp, heating_power, speed):
    return current_temp + heating_power * time_step / speed

# 시뮬레이션 수행
for t in range(1, conveyor_length):
    # 제품 위치 업데이트
    product_positions.append(product_positions[t-1] + speed[t-1] * time_step)
    times.append(t * time_step)
    
    current_temp = natural_temperature + (actual_temperatures[-1]-natural_temperature) * np.exp(-natural_temperature_k * time_step) 
    
    if product_positions[t] >= conveyor_length:
        product_positions[t] = conveyor_length  # 끝에 도달하면 멈춤
        actual_temperatures.append(current_temp)
        break

    heating_power = 0

    # 가열 구역 확인 및 온도 조정
    for zone in heating_zones:
        heating_power += heating_power_coef * np.exp(-0.5 * (product_positions[t] - zone / heating_power_std) ** 2)
    
    current_temp = adjust_heating_setting(current_temp, heating_power,speed[t])
    actual_temperatures.append(current_temp)

# 시뮬레이션 종료 시점 맞추기
end_time = len(product_positions)

# 결과 CSV 파일로 저장
data = {
    'Time (s)': times[:end_time],
    'Product Position': product_positions[:end_time],
    'Actual Temperature': actual_temperatures[:end_time]
}

df = pd.DataFrame(data)
df.to_csv('./simulation_results.csv', index=False)

# 결과 시각화
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(times[:end_time], speed[:end_time], label='Product Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed')
plt.title('Product Speed over Time')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(times[:end_time], product_positions[:end_time], label='Product Position', color='purple')
for zone in heating_zones:
    plt.axvline(zone, color='orange', linestyle='--', label=f'Heating Zone {zone}')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Product Position over Time')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(times[:end_time], actual_temperatures[:end_time], label='Actual Temperature', color='red')
plt.axhline(target_temperature, color='green', linestyle='--', label='Target Temperature')
plt.axhline(target_temperature + temperature_tolerance, color='blue', linestyle='--', label='Upper Tolerance')
plt.axhline(target_temperature - temperature_tolerance, color='blue', linestyle='--', label='Lower Tolerance')
for zone in heating_zones:
    plt.axvline(zone, color='orange', linestyle='--', label=f'Heating Zone {zone}')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Control Simulation')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Simulation results have been saved to 'simulation_results.csv'")