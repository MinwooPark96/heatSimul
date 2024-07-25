import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simulater:
    def __init__(self,
                time_step: int = 1,
                target_temperature: float = 200.0,
                initial_temperature: float = 200.0,
                tolerance: float = 10.0,
                conveyor_length: int = 1000,
                newton_cooling_coefficient: float = 0.001,
                natural_temperature: float = 25.0,
                seed: int = 42,
                mean_conveyor_speed: float = 2.0
                ):
    

        self.time_step = time_step
        self.target_temperature = target_temperature
        self.initial_temperature = initial_temperature
        self.tolerance = tolerance
        self.conveyor_length = conveyor_length
        self.newton_cooling_coefficient = newton_cooling_coefficient
        self.natural_temperature = natural_temperature
        self.seed = seed
        self.mean_conveyor_speed = mean_conveyor_speed
        
    def generate_speed_of_converyor(self):        
        return np.random.normal(self.mean_conveyor_speed, 0.5)


    def next_temparature_by_heat(self, 
                         current_temparature: float, 
                         speed_of_converyor: float,
                         heating_power: float):
        
        return current_temparature + heating_power * self.time_step / speed_of_converyor
    
    def next_temparature_by_natural_cooling(self,
                                            current_temparature: float,
                                            speed_of_converyor: float):
        # return current_temparature - self.newton_cooling_coefficient * (current_temparature - self.natural_temperature) * self.time_step / speed_of_converyor
        return self.natural_temperature\
            + (current_temparature - self.natural_temperature)\
                  * np.exp(-self.newton_cooling_coefficient * self.time_step)

    def simulate(self, 
                heating_power_coef: list[float],
                heating_point: list[int]
                   ):
        np.random.seed(self.seed)
        heating_power_std = [p/2 for p in heating_power_coef]
        # 초기화
        actual_temperatures = [np.float64(self.initial_temperature)]
        product_positions = [0]
        times = [0]
        speed_history = [] #[self.generate_speed_of_converyor()]

        for t in range(1, self.conveyor_length):
            speed_history.append(self.generate_speed_of_converyor())
            product_positions.append(product_positions[t-1] + speed_history[t-1] * self.time_step)
            times.append(t * self.time_step)
            
            natural_temp = self.next_temparature_by_natural_cooling(actual_temperatures[t-1], speed_history[t-1])
            
            if product_positions[t] >= self.conveyor_length:
                product_positions[t] = self.conveyor_length  
                speed_history.append(self.mean_conveyor_speed)
                actual_temperatures.append(natural_temp)
                break

            heating_power = 0

            # 가열 구역 확인 및 온도 조정
            for idx,zone in enumerate(heating_point):
                up = -(product_positions[t] - zone)**2 
                down = (2 * heating_power_std[idx])**2
                heating_power += heating_power_coef[idx] * np.exp( up / down)

            heated_temp = self.next_temparature_by_heat(natural_temp,speed_history[t-1],heating_power)
            actual_temperatures.append(heated_temp)

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

        return df, actual_temperatures[:end_time], product_positions[:end_time], times[:end_time], speed_history[:end_time]
    

    def draw_plot_with_simulator(self,
                heating_power_coef: list[float],
                heating_point: list[int]):
        
        _ , actual_temperatures, product_positions, times, speed_history = self.simulate(heating_power_coef, heating_point)

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(times, speed_history, label='Product Speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed')
        plt.title('Product Speed over Time')
        plt.legend()

        plt.subplot(2, 1, 2)
        target_temperature = 200
        temperature_tolerance = 10
        plt.plot(product_positions, actual_temperatures, label='Actual Temperature', color='red')
        plt.axhline(target_temperature, color='green', linestyle='--', label='Target Temperature')
        plt.axhline(target_temperature + temperature_tolerance, color='blue', linestyle='--', label='Upper Tolerance')
        plt.axhline(target_temperature - temperature_tolerance, color='blue', linestyle='--', label='Lower Tolerance')
        for zone in heating_point:  
            plt.axvline(zone, color='orange', linestyle='--', label=f'Heating Zone {zone}')
        plt.xlabel('Position (cm)')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Control Simulation')
        plt.legend()

        plt.tight_layout()
        plt.show()

        
if __name__ == '__main__':
    simulater = Simulater()
    df, *_ = simulater.simulate(heating_power_coef=100, heating_point=[20, 40, 60, 80])