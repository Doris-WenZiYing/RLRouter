import math
from collections import defaultdict
from itertools import combinations

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

def get_sfs_at_roadm(roadm_id, topology, current_sf_list):
    connected_sf_names = [sf[0] for sf in topology.get(roadm_id, {}).get('ConnectedSF', [])]
    return [sf for sf in current_sf_list if sf[0] in connected_sf_names]

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
                if total_delay > T_limit:
                    continue
                if total_delay < delay_dict[neighbor][0]:
                    delay_dict[neighbor] = (total_delay, current)
                    next_queue.append((neighbor, total_delay))
        queue = next_queue
    return delay_dict

def delay_sf_allocation(Tmax_list, Task_list, ROADM_list, Wavelength_list, SF_list, Topology):
    global Success, Fail
    NodeDelay = 0.0001
    lambda_assignment = {}
    original_sf_list = list(SF_list)  # 保存原始 SF 列表

    for i in range(len(Task_list)):
        task_id, host_id, Di, Fi, Mi = Task_list[i]
        Tmax = Tmax_list[i]
        current_task_id = task_id
        current_sf_list = list(SF_list) # 使用當前可用的 SF 列表

        print(f"\n[Task {task_id}] Mi (required memory): {Mi}")

        if not current_sf_list:
            print(f"Task {task_id}: No available SF left")
            Fail += 1
            continue

        compute_min = min(sf[1] for sf in current_sf_list)
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

        delay_dict = bfs_delay_expansion(host_roadm, Topology, ROADM_list, T_limit, initial_delay)

        feasible_allocations = []

        for i in range(1, len(current_sf_list) + 1):
            for selected_sfs_tuple in combinations(current_sf_list, i):
                selected_sfs = list(selected_sfs_tuple)
                total_memory = sum(sf[3] for sf in selected_sfs)
                if total_memory >= Mi:
                    total_compute_capacity = sum(sf[1] / sf[4] for sf in selected_sfs)
                    processing_time = Fi / total_compute_capacity if total_compute_capacity > 0 else float('inf')

                    max_network_delay = 0
                    valid_combination = True

                    for sf in selected_sfs:
                        sf_roadm = find_roadm_for_sf(sf, Topology)
                        if sf_roadm:
                            sf_delay = delay_dict.get(sf_roadm, (math.inf, None))[0]
                            if sf_delay == math.inf:
                                valid_combination = False
                                break
                            max_network_delay = max(max_network_delay, sf_delay)
                        else:
                            valid_combination = False
                            break

                    if valid_combination:
                        total_task_delay = max_network_delay + processing_time
                        if total_task_delay <= Tmax:
                            total_power = sum(sf[2] for sf in selected_sfs)
                            best_execution_roadm = min(delay_dict, key=delay_dict.get) if delay_dict else host_roadm
                            path = path_from_host_to_roadm(host_roadm, best_execution_roadm, delay_dict)
                            cost = 1.0 * total_task_delay + 0.01 * total_power
                            feasible_allocations.append({
                                'roadm': best_execution_roadm,
                                'delay': max_network_delay,
                                'selected_sfs': selected_sfs,
                                'processing_time': processing_time,
                                'total_task_delay': total_task_delay,
                                'total_power': total_power,
                                'path': path,
                                'cost': cost
                            })

        if not feasible_allocations:
            print(f"Task {task_id}: No feasible SF combination found within the constraints.")
            Fail += 1
            continue

        print(f"\n[Task {task_id}] All feasible delay+SF combinations:")
        for alloc in feasible_allocations:
            print(f"    ROADM={alloc['roadm']}, delay={alloc['delay']:.9f}, SFs={[sf[0] for sf in alloc['selected_sfs']]}, processing_time={alloc['processing_time']:.9f}, total_delay={alloc['total_task_delay']:.9f}, cost={alloc['cost']:.9f}, path={alloc['path']}")

        feasible_allocations.sort(key=lambda x: x['cost'])
        best_allocation = feasible_allocations[0]

        print(f"\n✅ Task {task_id} completed (Best option)")
        print("    Selected ROADM:", best_allocation['roadm'])
        print("    Selected SFs:", [sf[0] for sf in best_allocation['selected_sfs']])
        print("    Processing time:", best_allocation['processing_time'])
        print("    Total network delay:", best_allocation['delay'])
        print("    Total task delay:", best_allocation['total_task_delay'])
        print("    Total power consumption:", best_allocation['total_power'])
        print("    Path:", best_allocation['path'])
        print("    Cost:", best_allocation['cost'])
        Success += 1

        for sf in best_allocation['selected_sfs']:
            if sf not in SF_list:
                SF_list.append(sf)

        final_path = best_allocation['path']
        for j in range(len(final_path) - 1):
            u, v = final_path[j], final_path[j + 1]
            forward_lambdas = Wavelength_list.get((u, v), [])
            backward_lambdas = Wavelength_list.get((v, u), [])
            forward_lambda = random_select(forward_lambdas)
            backward_lambda = random_select(backward_lambdas)
            if forward_lambda:
                Wavelength_list[(u, v)].remove(forward_lambda)
                lambda_assignment[(current_task_id, u, v)] = forward_lambda
                print(f"    🔄 Assigned λ={forward_lambda} to ({u}, {v})")
            if backward_lambda:
                Wavelength_list[(v, u)].remove(backward_lambda)
                lambda_assignment[(current_task_id, v, u)] = backward_lambda
                print(f"    🔄 Assigned λ={backward_lambda} to ({v}, {u})")

        for (tid, r1, r2), lamb in list(lambda_assignment.items()):
            if tid == current_task_id:
                if (r1, r2) not in Wavelength_list:
                    Wavelength_list[(r1, r2)] = []
                Wavelength_list[(r1, r2)].append(lamb)
                print(f"    🔄 Returned λ={lamb} to ({r1}, {r2})")
                del lambda_assignment[(tid, r1, r2)]



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