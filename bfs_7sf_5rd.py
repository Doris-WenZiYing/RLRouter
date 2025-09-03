import math
from collections import defaultdict

def calculate_propagation_delay(distance):
    return distance / (2 * 10**5)

def random_select(wavelengths):
    return wavelengths[0] if wavelengths else None

def find_roadm_for_host(host_id, topology):
    for roadm, info in topology.items():
        if host_id in info.get('ConnectedHosts', []):
            return roadm
    return None

def find_roadm_for_sf(sf_entry, topology):
    for roadm, info in topology.items():
        if sf_entry in info.get('ConnectedSF', []):
            return roadm
    return None

def path_from_host_to_roadm(start, end, delay_dict):
    path = []
    current = end
    while current != start:
        path.insert(0, current)
        current = delay_dict[current][1]
    path.insert(0, start)
    return path

def bfs_delay_expansion(host_roadm, Topology, ROADM_list, T_limit, initial_delay):
    delay_dict = {roadm: (math.inf, None) for roadm in Topology}
    delay_dict[host_roadm] = (initial_delay, None)
    queue = [(host_roadm, initial_delay)]

    while queue:
        next_queue = []
        for current, current_delay in queue:
            for neighbor in Topology.get(current, {}).get('Neighbors', []):
                if neighbor not in Topology:
                    continue
                distance = ROADM_list.get((current, neighbor))
                if distance is None:
                    continue
                Tprop = calculate_propagation_delay(distance)
                total_delay = current_delay + Tprop + 0.0001
                print(f"From {current} → {neighbor}:")
                print(f"  Current delay: {current_delay}")
                print(f"  Propagation: {Tprop}")
                print(f"  Total delay: {total_delay}")
                print(f"  Existing delay in dict: {delay_dict[neighbor][0]}")
                if total_delay > T_limit:
                    continue
                if total_delay <= delay_dict[neighbor][0]:
                    delay_dict[neighbor] = (total_delay, current)
                    next_queue.append((neighbor, total_delay))
        queue = next_queue
    return delay_dict

def delay_sf_allocation(Tmax_list, Task_list, ROADM_list, Wavelength_list, SF_list, Topology, collector=None):
    global Success, Fail
    NodeDelay = 0.0001
    lambda_assignment = {}

    for i in range(len(Task_list)):
        task_id, host_id, Di, Fi, Mi = Task_list[i]
        Tmax = Tmax_list[i]
        current_task_id = task_id

        print(f"\n[Task {task_id}] Mi (required memory): {Mi}")

        if not SF_list:
            print(f"Task {task_id}: No available SF left")
            Fail += 1
            continue

        compute_min = min(sf[1] for sf in SF_list)
        Tcompute = Fi / compute_min
        print(f"[Task {task_id}] Tcompute = {Tcompute:.6f}, Tmax = {Tmax}")

        if Tmax <= Tcompute:
            print(f"Task {task_id} cannot be completed: Tcompute > Tmax")
            Fail += 1
            continue

        T_limit = Tmax - Tcompute
        print(f"[Task {task_id}] T_limit = Tmax - Tcompute = {T_limit:.6f}")

        host_roadm = find_roadm_for_host(host_id, Topology)
        if host_roadm is None:
            print(f"Task {task_id}: Host not connected to ROADM")
            Fail += 1
            continue

        B = 50 * 10**9
        gOSNR = 20
        Ttrans = Di / (B * math.log2(1 + gOSNR))
        initial_delay = Ttrans + NodeDelay
        print(f"    ↪ Ttrans (for Di={Di}): {Ttrans:.9f} sec")
        print(f"    ↪ Initial delay: {initial_delay:.9f} sec")

        delay_dict = bfs_delay_expansion(host_roadm, Topology, ROADM_list, T_limit, initial_delay)

        Delay_List = []
        for roadm in delay_dict:
            delay, prev = delay_dict[roadm]
            if delay <= T_limit:
                for sf in SF_list:
                    if find_roadm_for_sf(sf, Topology) == roadm:
                        Delay_List.append((roadm, delay, sf, sf[1], sf[2], sf[3], sf[4]))

        if not Delay_List:
            print(f"Task {task_id}: Cannot be assigned")
            Fail += 1
            continue

        Delay_List.sort(key=lambda x: x[1])
        print(f"\n[Task {task_id}] Available delay+SF 組合:")
        for entry in Delay_List:
            print(f"    ROADM={entry[0]}, delay={entry[1]:.9f}, SF={entry[2][0]}, compute={entry[3]}, power={entry[4]}, mem={entry[5]}")

        selected_SF = []
        accumulated_compute_capacity = 0
        accumulated_memory = 0
        total_power_consumption = 0
        previous_cost = float('inf')
        alpha = 1.0
        beta = 0.01

        for ROADM, delay, SF, compute_power, power, mem, threads in Delay_List:
            selected_SF.append(SF)
            accumulated_compute_capacity += compute_power / threads
            accumulated_memory += mem
            total_power_consumption += power

            print(f"      ➤ 加入 SF={SF[0]} 時累積 mem = {accumulated_memory} / {Mi}")

            if accumulated_memory < Mi:
                continue

            processing_time = Fi / accumulated_compute_capacity
            path = path_from_host_to_roadm(host_roadm, ROADM, delay_dict)
            cost = alpha * (delay + processing_time) + beta * total_power_consumption
            print("  ➤ Path:", path)
            print(f"    ➤ Testing combo: SF={SF[0]}, delay={delay:.6f}, processing_time={processing_time:.6f}, cost={cost:.6f}")

            if cost < previous_cost:
                final_delay = delay
                final_selected_sf = list(selected_SF)
                final_processing_time = processing_time
                previous_cost = cost
                final_cost = cost
                final_path = list(path)
            else:
                selected_SF.pop()
                accumulated_compute_capacity -= compute_power / threads
                accumulated_memory -= mem
                total_power_consumption -= power
                break

        if accumulated_memory < Mi:
            print(f"Task {task_id}: ❌ Not enough memory ({accumulated_memory} < {Mi})")
            Fail += 1
            continue

        print(f"\n✅ Task {task_id} completed")
        print("  Selected SF:", final_selected_sf)
        Success += 1
        print("  Processing time:", final_processing_time)
        print("  Total network delay:", final_delay)
        print("  Total task delay:", final_processing_time + final_delay)
        print("  Total power consumption:", total_power_consumption)
        print("  Cost:", final_cost)

        if collector is not None:
            collector.setdefault('delays', []).append(final_processing_time + final_delay)
            collector.setdefault('powers', []).append(total_power_consumption)

        for sf in final_selected_sf:
            if sf in SF_list:
                SF_list.remove(sf)

        for i in range(len(final_path) - 1):
            u, v = final_path[i], final_path[i + 1]
            forward_lambdas = Wavelength_list.get((u, v), [])
            backward_lambdas = Wavelength_list.get((v, u), [])
            forward_lambda = random_select(forward_lambdas)
            backward_lambda = random_select(backward_lambdas)
            if forward_lambda:
                Wavelength_list[(u, v)].remove(forward_lambda)
                lambda_assignment[(current_task_id, u, v)] = forward_lambda
                print(f"  🔄 Assigned λ={forward_lambda} to ({u}, {v})")
            if backward_lambda:
                Wavelength_list[(v, u)].remove(backward_lambda)
                lambda_assignment[(current_task_id, v, u)] = backward_lambda
                print(f"  🔄 Assigned λ={backward_lambda} to ({v}, {u})")

        for (tid, r1, r2), lamb in list(lambda_assignment.items()):
            if tid == current_task_id:
                if (r1, r2) not in Wavelength_list:
                    Wavelength_list[(r1, r2)] = []
                Wavelength_list[(r1, r2)].append(lamb)
                print(f"  🔄 Returned λ={lamb} to ({r1}, {r2})")
                del lambda_assignment[(tid, r1, r2)]

        for sf in final_selected_sf:
            if sf not in SF_list:
                SF_list.append(sf)

# === Initialization ===
Success = 0
Fail = 0

GPU_catalog = {
    'A100': [1e8, 40],
    'H100': [2e8, 80],
}


SF_list = [
    ('SF1', GPU_catalog['A100'][0], 30, GPU_catalog['A100'][1], 1),  # r1
    ('SF2', GPU_catalog['H100'][0], 30, GPU_catalog['H100'][1], 1),  # r2
    ('SF3', GPU_catalog['A100'][0], 20, GPU_catalog['A100'][1], 1),  # r3
    ('SF4', GPU_catalog['A100'][0], 30, GPU_catalog['A100'][1], 1),  # r1
    ('SF5', GPU_catalog['H100'][0], 20, GPU_catalog['H100'][1], 1),  # r4
    ('SF6', GPU_catalog['A100'][0], 25, GPU_catalog['A100'][1], 1),  # r3 (新增)
    ('SF7', GPU_catalog['H100'][0], 15, GPU_catalog['H100'][1], 1),  # r5 (新增)
]

Tmax_list = [1.5, 3, 5, 3.5, 4]
Task_list = [
    (1, 'h1', 8e6, 1e8, 40),     # r1
    (2, 'h2', 16e6, 2e8, 80),    # r2
    (3, 'h3', 80e6, 4e8, 120),   # r3
    (4, 'h4', 32e6, 3e8, 100),   # r4
    (5, 'h5', 64e6, 5e8, 160),   # r5
]

ROADM_list = {
    ('r1', 'r2'): 100,
    ('r2', 'r1'): 100,
    ('r2', 'r3'): 150,
    ('r3', 'r2'): 150,
    ('r3', 'r4'): 200,
    ('r4', 'r3'): 200,
    ('r3', 'r5'): 250,
    ('r5', 'r3'): 250,
}

Wavelength_list = {
    ('r1', 'r2'): ['λ1', 'λ2'],
    ('r2', 'r1'): ['λ3', 'λ4'],
    ('r2', 'r3'): ['λ5', 'λ6'],
    ('r3', 'r2'): ['λ7', 'λ8'],
    ('r3', 'r4'): ['λ9', 'λ10'],
    ('r4', 'r3'): ['λ11', 'λ12'],
    ('r3', 'r5'): ['λ13', 'λ14'],
    ('r5', 'r3'): ['λ15', 'λ16'],
}

Topology = {
    'r1': {
        'Neighbors': ['r2'],
        'ConnectedHosts': ['h1'],
        'ConnectedSF': [SF_list[0], SF_list[3]],
    },
    'r2': {
        'Neighbors': ['r1', 'r3'],
        'ConnectedHosts': ['h2'],
        'ConnectedSF': [SF_list[1]],
    },
    'r3': {
        'Neighbors': ['r2', 'r4', 'r5'],
        'ConnectedHosts': ['h3'],
        'ConnectedSF': [SF_list[2], SF_list[5]],
    },
    'r4': {
        'Neighbors': ['r3'],
        'ConnectedHosts': ['h4'],
        'ConnectedSF': [SF_list[4]],
    },
    'r5': {
        'Neighbors': ['r3'],
        'ConnectedHosts': ['h5'],
        'ConnectedSF': [SF_list[6]],
    },
}

delay_sf_allocation(Tmax_list, Task_list, ROADM_list, Wavelength_list, SF_list, Topology)

total = Success + Fail
success_rate = Success / total if total > 0 else 0
print(f"\n📊 統計結果：")
print(f"  ✔ 成功任務數：{Success}")
print(f"  ❌ 失敗任務數：{Fail}")
print(f"  ✅ 成功率：{success_rate * 100:.2f}%")