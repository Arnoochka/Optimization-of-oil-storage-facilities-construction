import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Задаём параметры задачи
n_storages = 3  # Количество хранилищ
m_locations = 100  # Количество возможных мест
oil_types = ['A', 'B', 'C']  # Типы нефти
oil_prices = {'A': 100, 'B': 120, 'C': 150}  # Цена за единицу нефти
storage_cost = 10  # Стоимость хранения за единицу
transport_cost_per_km = 2  # Стоимость транспортировки за 1 км
transport_time_factor = {'A': 1.2, 'B': 1.5, 'C': 1.8}  # Коэффициент времени на транспортировку

# Случайное размещение точек (x, y)
np.random.seed(42)
locations = np.random.rand(m_locations, 2) * 100  # Координаты мест
supply_point = np.array([50, 50])  # Точка сбыта

# Создание уникального индивидуума
def create_unique_individual():
    return random.sample(range(m_locations), n_storages)

# Функция оценки (fitness)
def calculate_fitness(individual):
    # Выбираем места для хранилищ
    selected_locations = [locations[i] for i in individual]
    
    # Случайное распределение типов нефти по хранилищам
    assigned_types = random.sample(oil_types * (n_storages // len(oil_types)) + oil_types[:n_storages % len(oil_types)], n_storages)
    
    # Считаем прибыль и затраты
    profit = 0
    storage_expense = n_storages * storage_cost
    transport_expense = 0
    
    for i, loc in enumerate(selected_locations):
        oil_type = assigned_types[i]
        price = oil_prices[oil_type]
        
        # Расстояние до точки сбыта
        distance = np.linalg.norm(loc - supply_point)
        transport_expense += distance * transport_cost_per_km * transport_time_factor[oil_type]
        profit += price  # Упрощённая прибыль только от продажи нефти
    
    total_fitness = profit - storage_expense - transport_expense
    return total_fitness, assigned_types  # Возвращаем также назначенные типы нефти

# Мутация с сохранением уникальности
def unique_mutation(individual):
    idx_to_mutate = random.randint(0, len(individual) - 1)  # Выбираем индекс для мутации
    available_positions = list(set(range(m_locations)) - set(individual))  # Доступные места
    if available_positions:  # Если остались свободные места
        individual[idx_to_mutate] = random.choice(available_positions)
    return individual,

# Настройка генетического алгоритма
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_unique_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", unique_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", calculate_fitness)

# Запуск алгоритма
population = toolbox.population(n=50)
n_generations = 50
fitness_values = []

for gen in range(n_generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    
    # Оцениваем пригодность потомков
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit[0],  # Теперь fit - это кортеж, и мы сохраняем только первое значение
    
    # Отбор для следующего поколения
    population = toolbox.select(offspring, k=len(population))
    fitness_values.append(max(ind.fitness.values[0] for ind in population))

best_ind = tools.selBest(population, k=1)[0]

# Получаем типы нефти для выбранных хранилищ
best_fitness, assigned_types = calculate_fitness(best_ind)

# Вывод всех параметров итогового решения
print(f"Итоговое решение:")
print(f"Выбранные хранилища (места): {best_ind}")
print(f"Назначенные типы нефти: {assigned_types}")

# Визуализация
def plot_solution(best_individual, assigned_types):
    selected_locations = [locations[i] for i in best_individual]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(locations[:, 0], locations[:, 1], c='blue', label='Potential Locations')
    plt.scatter(supply_point[0], supply_point[1], c='red', label='Supply Point', s=100, marker='X')
    for i, loc in enumerate(selected_locations):
        plt.scatter(loc[0], loc[1], s=100, label=f'Storage {assigned_types[i]}', marker='o')
        plt.plot([loc[0], supply_point[0]], [loc[1], supply_point[1]], c='gray', linestyle='--')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title('Optimal Storage and Routes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# Отображаем решение
plot_solution(best_ind, assigned_types)

# График прогресса
plt.plot(fitness_values)
plt.title('Fitness Progress')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
