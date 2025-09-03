
import math
import heapq  #thread,memory,gpu幾核  #比較最佳解(可暴力法)   #找ai routing模型

def calculate_propagation_delay(distance):
    return distance / (2 * 10**5)       #2*10^5km/s

def random_select(wavelengths):
    return wavelengths[0] if wavelengths else None 

def find_roadm_for_host(host_id, topology):
    for roadm, info in topology.items():
        if host_id in info.get('ConnectedHosts', []):
            return roadm
    return None

def path_from_host_to_roadm(start, end, delay_dict):
    path = []
    current = end
    while current != start:
        path.insert(0, current)
        current = delay_dict[current][3]
    path.insert(0, start)
    return path

def delay_sf_allocation(Tmax_list, Task_list, ROADM_list, Wavelength_list, SF_list, Topology):
    global Success, Fail
    NodeDelay = 0.0001
    lambda_assignment = {}

    for i in range(len(Task_list)):
        task_id, host_id, Di, Fi = Task_list[i]
        Tmax = Tmax_list[i]
        current_task_id = task_id

        if not SF_list:
            print(f"Task {task_id}: No available SF left")
            Fail += 1
            continue

        compute_min = min(sf[1] for sf in SF_list)
        Tcompute = Fi / compute_min
        print(f"\n[Task {task_id}] Tcompute = {Tcompute:.6f}, Tmax = {Tmax}")

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
        print(f"    ↪ Ttrans (for Di={Di}): {Ttrans:.9f} sec")
                
        initial_delay = Ttrans + NodeDelay
        print(f"    ↪ Initial delay: {initial_delay:.9f} sec")
        
        delay_dict = {roadm: (math.inf, 0, 0, None) for roadm in Topology}
        pq = [(initial_delay, host_roadm, 1, 0, None)]
        delay_dict[host_roadm] = (initial_delay, 1, 0, None)

        while pq:
            current_delay, roadm, roadm_count, total_distance, prev_roadm = heapq.heappop(pq)

            for neighbor in Topology.get(roadm, {}).get('Neighbors', []):
                if neighbor == prev_roadm:
                    continue  # 不要走回頭路
                if neighbor not in Topology:
                    continue
                
                distance = ROADM_list.get((roadm, neighbor))
                if distance is None:
                    continue
                                       

                Tprop = calculate_propagation_delay(distance)
                print(f"    ↪ Tprop (from {roadm} to {neighbor}): {Tprop:.9f} sec")
                
                future_roadm_count = roadm_count + 1
                

                hop_delay = NodeDelay * (future_roadm_count + 2)
                print(f"    ↪ Hop delay (hop count = {future_roadm_count + 2}): {hop_delay:.9f} sec")


                if future_roadm_count == 2:
                    total_delay = current_delay + Tprop + (NodeDelay * (future_roadm_count + 1))
                else:
                    total_delay = current_delay + Tprop + NodeDelay
                    
                print(f"    ↪ Total delay to {neighbor}: {total_delay:.9f} sec")

                if total_delay > T_limit:
                    print(f"    ✘ Exceeds T_limit ({T_limit:.6f}), skipping {neighbor}")
                    continue

                if total_delay < delay_dict[neighbor][0]:
                    delay_dict[neighbor] = (total_delay, roadm_count + 1, total_distance + distance, roadm)
                    heapq.heappush(pq, (total_delay, neighbor, roadm_count + 1, total_distance + distance, roadm))

        Delay_List = []
        for roadm in delay_dict:
            if delay_dict[roadm][0] <= T_limit:
                for sf in Topology[roadm].get('ConnectedSF',[]):
                    Delay_List.append((roadm, delay_dict[roadm][0], delay_dict[roadm][2], sf, sf[1], sf[2]))

        if not Delay_List:
            print(f"Task {task_id}: Cannot be assigned")
            Fail += 1
            continue

        Delay_List.sort(key=lambda x: x[1])
        print("\n[Task", task_id, "] Available delay+SF 組合:")
        for entry in Delay_List:
            print(f"    ROADM={entry[0]}, delay={entry[1]:.9f}, SF={entry[3][0]}, compute={entry[4]}, power={entry[5]}")
        selected_SF = []
        accumulated_compute_capacity = 0
        total_power_consumption = 0
        previous_cost = float('inf')
        alpha = 1.0
        beta = 0.01

        for ROADM, delay, total_distance, SF, compute_power, power_consumption in Delay_List:
            selected_SF.append(SF)
            accumulated_compute_capacity += compute_power
            total_power_consumption += power_consumption
            processing_time = Fi / accumulated_compute_capacity
            total_task_delay = delay + processing_time
            path = path_from_host_to_roadm(host_roadm, ROADM, delay_dict)
            print("  ➤ Path:", path)

            cost = alpha * total_task_delay + beta * total_power_consumption
            
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
                accumulated_compute_capacity -= compute_power
                total_power_consumption -= power_consumption
                break

        print(f"\n✅ Task {task_id} completed")
        print("  Selected SF:", final_selected_sf)
        Success += 1
        print("  Processing time:", final_processing_time)
        print("  Total network delay:", final_delay)
        print("  Total task delay:", final_processing_time+final_delay)
        print("  Total power consumption:", total_power_consumption)
        print("  Cost:", final_cost)

        # 將已分配的 SF 從可用清單中移除
        for sf in selected_SF:
            if sf in SF_list:
                SF_list.remove(sf)

        for i in range(len(final_path) - 1):
            u, v = final_path[i], final_path[i + 1]

            # 把 forward 與 backward 都抓出來
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
        


if __name__ == '__main__':
    # 變數清單註解：
    # - Tmax_list: List[float]，每個任務可用最大總時間（秒）
    # - Task_list: List[Tuple[int, str, int, int]] = (task_id, host_id, Di, Fi)
    # - SF_list: List[Tuple[str, int, int]] = (sf_id, compute_power, power_consumption)
    # - ROADM_list: Dict[Tuple[str, str], float] = link距離
    # - Wavelength_list: Dict[Tuple[str, str], List[str]] = 每條link上的lambda列表
    # - Topology: Dict[str, Dict] = 拓撲圖資訊，包含鄰居與host連線

    Fail = 0
    Success = 0
    Tmax_list = [1.5, 3, 5]
    Task_list = [
        (1, 'h1', 8e6, 1e8), #1MB = 8&10^6bits
        (2, 'h2', 16e6, 2e8),
        (3, 'h3', 80e6, 4e8),
    ]
    SF_list = [
        ('SF1', 1e8, 30),
        ('SF2', 1e8, 30),
        ('SF3',2e8, 20)
    ]
    ROADM_list = {
        ('r1', 'r2'): 200, #km
        ('r2', 'r1'): 200,
        ('r2', 'r3'): 300,
        ('r3', 'r2'): 300,
    }
    Wavelength_list = {
        ('r1', 'r2'): ['λ1', 'λ2'],
        ('r2', 'r3'): ['λ5', 'λ6'],
        ('r2', 'r1'): ['λ3', 'λ4'],
        ('r3', 'r2'): ['λ7', 'λ8']
    }
    Topology = {
        'r1': {
            'Neighbors': ['r2'],
            'ConnectedHosts': ['h1'],
            'ConnectedSF':[('SF1', 1e8, 30)]
        },
        'r2': {
            'Neighbors': ['r1', 'r3'],
            'ConnectedHosts': ['h2'],
            'ConnectedSF':[('SF2', 1e9, 30)]
        },
        'r3': {
            'Neighbors': ['r2'],
            'ConnectedHosts': ['h3'],
            'ConnectedSF':[('SF3',5e9, 20)]
        }
    }

    delay_sf_allocation(Tmax_list, Task_list, ROADM_list, Wavelength_list, SF_list, Topology)

    total = Fail + Success
    success_rate = Success / total if total > 0 else 0
    print(f"\n📊 統計結果：")
    print(f"  ✔ 成功任務數：{Success}")
    print(f"  ❌ 失敗任務數：{Fail}")
    print(f"  ✅ 成功率：{success_rate * 100:.2f}%")
