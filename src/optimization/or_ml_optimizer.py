import os
import sys
import time
import copy
import random
from typing import List, Dict, Tuple, Optional, Any

# Import OR-Tools
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from src.models.route import Route
from src.models.location import Location
from src.models.parcel import Parcel
from src.optimization.base_optimizer import BaseOptimizer

class ORToolsMLOptimizer(BaseOptimizer):
    """
    OR-Tools with ML optimizer for VRP.
    This is Method 2 in the four-method system.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the OR-Tools with ML optimizer.
        
        Args:
            data_processor: DataProcessor containing problem data
        """
        super().__init__(data_processor)
        self._location_cache = {}
        
        # Initialize warehouse location
        self.warehouse_location = self._get_warehouse_location()
        
        # ML model will be loaded or trained lazily
        self.route_predictor = None
        self._ortools_solutions = []
        
    def optimize(self) -> List[Route]:
        """
        Execute the OR-Tools with ML optimization.
        
        Returns:
            List[Route]: The best solution found
        """
        print("\nStarting OR-Tools with ML optimization...")
        start_time = time.time()
        
        # Ensure ML model is loaded
        self._ensure_ml_model_loaded()
        
        # Generate initial OR-Tools solutions
        or_solutions = self._generate_or_tools_solutions(num_variations=5)
        self._ortools_solutions = [route for solution in or_solutions for route in solution]
        
        # Use ML to enhance solutions
        enhanced_solutions = self._enhance_with_ml(self._ortools_solutions)
        
        # Evaluate all solutions with detailed metrics
        all_solutions = []
        for solution in enhanced_solutions:
            if not solution:  # Skip empty solutions
                continue
                
            evaluation = self.evaluate_solution(solution)
            
            # Calculate normalized scores (0-1 range where 1 is better)
            distance_score = 1.0 / (1.0 + evaluation['total_distance'])  # Lower distance is better
            cost_score = 1.0 / (1.0 + evaluation['total_cost'])  # Lower cost is better
            capacity_score = evaluation['capacity_utilization']  # Higher utilization is better
            delivery_score = evaluation['delivery_ratio']  # Higher delivery ratio is better
            efficiency_score = evaluation['route_efficiency'] / (1.0 + evaluation['route_efficiency'])  # Normalize efficiency
            
            # Calculate weighted composite score
            composite_score = (
                0.25 * distance_score +     # 25% weight on distance
                0.25 * cost_score +         # 25% weight on cost
                0.20 * capacity_score +     # 20% weight on capacity utilization
                0.20 * delivery_score +     # 20% weight on delivery ratio
                0.10 * efficiency_score     # 10% weight on route efficiency
            )
            
            print(f"\nSolution Evaluation:")
            print(f"- Distance Score: {distance_score:.4f} (from {evaluation['total_distance']:.1f} km)")
            print(f"- Cost Score: {cost_score:.4f} (from ${evaluation['total_cost']:.2f})")
            print(f"- Capacity Score: {capacity_score:.4f} ({evaluation['capacity_utilization']:.1%} utilization)")
            print(f"- Delivery Score: {delivery_score:.4f} ({evaluation['parcels_delivered']} parcels)")
            print(f"- Efficiency Score: {efficiency_score:.4f} ({evaluation['route_efficiency']:.2f} parcels*weight/km)")
            print(f"- Composite Score: {composite_score:.4f}")
            
            all_solutions.append((solution, composite_score))
        
        # Select best solution
        if all_solutions:
            all_solutions.sort(key=lambda x: x[1], reverse=True)  # Sort by composite score
            self.best_solution = all_solutions[0][0]
            best_score = all_solutions[0][1]
            
            # Print detailed evaluation of best solution
            best_eval = self.evaluate_solution(self.best_solution)
            print(f"\nBest Solution Details:")
            print(f"- Total Distance: {best_eval['total_distance']:.2f} km")
            print(f"- Total Cost: ${best_eval['total_cost']:.2f}")
            print(f"- Parcels Delivered: {best_eval['parcels_delivered']} ({best_eval['delivery_ratio']:.1%} of total)")
            print(f"- Capacity Utilization: {best_eval['capacity_utilization']:.1%}")
            print(f"- Route Efficiency: {best_eval['route_efficiency']:.2f} parcels*weight/km")
            print(f"- Composite Score: {best_score:.4f}")
        else:
            print("\nWarning: No valid solutions found")
            self.best_solution = []
            best_score = 0.0
        
        elapsed_time = time.time() - start_time
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        
        return self.best_solution
    
    def _ensure_ml_model_loaded(self):
        """Ensure the ML model is loaded or train a new one"""
        if self.route_predictor is None:
            # Import here to avoid circular imports
            from src.ml.route_predictor import RoutePredictor
            self.route_predictor = RoutePredictor()
            
            # Try to load existing model
            model_path = 'models/route_predictor.joblib'
            try:
                self.route_predictor.load_model(model_path)
                print("Loaded existing ML model")
            except:
                print("No existing ML model found, training new model...")
                # Generate a quick OR-Tools solution to train on
                initial_solutions = self._generate_or_tools_solutions(num_variations=3)
                all_routes = [route for solution in initial_solutions for route in solution]
                
                if all_routes:
                    # Convert to training data format
                    training_data = []
                    for route in all_routes:
                        route_data = {
                            'parcels': [{'destination': p.destination.city_name} for p in route.parcels],
                            'total_weight': route.get_total_weight(),
                            'total_distance': route.total_distance,
                            'vehicle_capacity': route.vehicle_capacity,
                            'truck_type': route.vehicle_id.split('_')[1]
                        }
                        training_data.append(route_data)
                    
                    # Train model
                    self.route_predictor.train(training_data)
                    
                    # Save model
                    os.makedirs('models', exist_ok=True)
                    self.route_predictor.save_model(model_path)
                    print(f"Trained new ML model with {len(training_data)} examples")
                else:
                    print("Warning: Could not generate training data")
    
    def _generate_or_tools_solutions(self, num_variations: int = 5) -> List[List[Route]]:
        """
        Generate multiple OR-Tools solutions with different parameters.
        
        Args:
            num_variations: Number of different solutions to generate
            
        Returns:
            List of solutions, where each solution is a list of routes
        """
        print("Generating OR-Tools solutions...")
        solutions = []
        
        # Vary parameters to get different solutions
        time_limits = [30, 60, 90]  # Increased time limits
        strategies = [
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES  # Added more strategies
        ]
        
        # Get initial data model to determine available truck types
        initial_data = self._create_data_model()
        max_vehicles = len(initial_data['truck_types'])
        
        # Calculate base number of vehicles from initial data
        base_vehicles = initial_data['num_vehicles']
        
        for i in range(num_variations):
            print(f"Generating solution variation {i+1}/{num_variations}")
            
            # Randomly select parameters
            time_limit = random.choice(time_limits)
            strategy = random.choice(strategies)
            
            # Try different vehicle counts around the base number
            vehicle_counts = [
                base_vehicles,  # Try base number
                max(1, base_vehicles - 1),  # Try one less
                min(base_vehicles + 1, max_vehicles),  # Try one more
                max(1, base_vehicles - 2),  # Try two less
                min(base_vehicles + 2, max_vehicles)   # Try two more
            ]
            
            # Try each vehicle count until we find a solution
            for new_num_vehicles in vehicle_counts:
                # Create data model with slight variations
                data = self._create_data_model()
                
                # Update truck types and capacities to match new vehicle count
                data['num_vehicles'] = new_num_vehicles
                data['truck_types'] = data['truck_types'][:new_num_vehicles]
                data['vehicle_capacities'] = [
                    float(self.data_processor.truck_specifications[t]['weight_capacity'] * 1000)
                    for t in data['truck_types']
                ]
                
                # Verify the lengths match before solving
                if len(data['vehicle_capacities']) != data['num_vehicles']:
                    continue
                
                # Solve with these parameters
                solution = self._solve_or_tools(data, strategy, time_limit)
                if solution:
                    solutions.append(solution)
                    print(f"Found solution with {new_num_vehicles} vehicles")
                    break
            
            # If we have enough solutions, stop
            if len(solutions) >= num_variations:
                break
        
        print(f"Generated {len(solutions)} OR-Tools solutions")
        return solutions
    
    def _create_data_model(self) -> Dict:
        """
        Create data model for OR-Tools solver.
        
        Returns:
            Dictionary with problem data
        """
        # Use all cities but limit distance matrix for efficiency
        cities = list(self.data_processor.cities)
        max_cities = 200  # Increased from 100
        if len(cities) > max_cities:
            cities = cities[:max_cities]
            distance_matrix = self.data_processor.distance_matrix[:max_cities, :max_cities]
        else:
            distance_matrix = self.data_processor.distance_matrix
        
        # Initialize data structures
        total_demand = 0
        demands = [0] * len(cities)  # Initialize with zeros
        order_to_node = {}
        order_locations = {}
        
        print("Processing orders for solver...")  # Debug print
        
        # Process orders - weights are already in tons from DataProcessor
        for _, row in self.data_processor.order_data.iterrows():
            dest = row['Destination']
            node_idx = self.data_processor.city_to_idx.get(dest, -1)
            if node_idx < len(cities) and node_idx != -1:
                weight = float(row['Weight'])  # Weight is already in tons
                demands[node_idx] += weight
                total_demand += weight
                order_to_node[row['Order_ID']] = node_idx
                order_locations[row['Order_ID']] = (row['Source'], dest)
        
        print(f"Total demand: {total_demand} tons")  # Debug print
        
        # Get available truck types and their capacities (already in tons)
        truck_types = sorted(list(self.data_processor.truck_specifications.keys()),
                           key=lambda t: self.data_processor.truck_specifications[t]['weight_capacity'])
        
        # Calculate required vehicles based on total demand and largest truck capacity
        max_truck_capacity = max(self.data_processor.truck_specifications[t]['weight_capacity'] 
                               for t in truck_types)
        min_vehicles_needed = max(1, int(total_demand / max_truck_capacity) + 1)
        
        # Set number of vehicles with a reasonable upper limit
        num_vehicles = min(20, min_vehicles_needed)
        print(f"Minimum vehicles needed: {min_vehicles_needed}, Using: {num_vehicles}")
        
        # Select appropriate truck types based on demand distribution
        selected_truck_types = []
        remaining_demand = total_demand
        for _ in range(num_vehicles):
            # Choose truck type based on remaining demand
            for truck_type in reversed(truck_types):  # Start with largest trucks
                capacity = self.data_processor.truck_specifications[truck_type]['weight_capacity']
                if capacity <= remaining_demand * 1.5:  # Allow some flexibility
                    selected_truck_types.append(truck_type)
                    remaining_demand -= capacity
                    break
            if not selected_truck_types or len(selected_truck_types) < num_vehicles:
                selected_truck_types.append(truck_types[0])  # Add smallest truck if needed
        
        # Get vehicle capacities for selected trucks (in tons)
        vehicle_capacities = [float(self.data_processor.truck_specifications[t]['weight_capacity']) 
                            for t in selected_truck_types[:num_vehicles]]
        
        # Ensure we have enough capacity
        total_capacity = sum(vehicle_capacities)
        if total_capacity < total_demand:
            print(f"Warning: Total capacity ({total_capacity} tons) is less than total demand ({total_demand} tons)")
            # Add more vehicles if needed
            while total_capacity < total_demand and len(vehicle_capacities) < 20:
                new_truck = max(truck_types, key=lambda t: self.data_processor.truck_specifications[t]['weight_capacity'])
                vehicle_capacities.append(float(self.data_processor.truck_specifications[new_truck]['weight_capacity']))
                selected_truck_types.append(new_truck)
                total_capacity = sum(vehicle_capacities)
                num_vehicles = len(vehicle_capacities)
        
        print(f"Final configuration:")
        print(f"- Number of vehicles: {num_vehicles}")
        print(f"- Total capacity: {total_capacity} tons")
        print(f"- Total demand: {total_demand} tons")
        print(f"- Capacity/Demand ratio: {total_capacity/total_demand if total_demand > 0 else 'inf'}")
        
        return {
            'distance_matrix': distance_matrix,
            'num_vehicles': num_vehicles,
            'demands': demands,
            'order_to_node': order_to_node,
            'order_locations': order_locations,
            'cities': cities,
            'depot': 0,
            'vehicle_capacities': vehicle_capacities,
            'truck_types': selected_truck_types[:num_vehicles]
        }
    
    def _solve_or_tools(self, data, strategy, time_limit: int) -> Optional[List[Route]]:
        """
        Solve VRP using OR-Tools with specified parameters.
        
        Args:
            data: Problem data
            strategy: First solution strategy
            time_limit: Time limit in seconds
            
        Returns:
            List of routes or None if no solution found
        """
        print("\nStarting OR-Tools solver...")
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(data['cities']), 
            data['num_vehicles'], 
            data['depot']
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Convert to integer and handle potential NaN values
            distance = data['distance_matrix'][from_node][to_node]
            if distance is None or distance != distance:  # Check for NaN
                return 0
            return int(distance * 100)  # Convert to integer (preserve 2 decimal places)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add distance dimension with more generous limits
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,    # no slack
            1000000,  # very large vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        
        # Define demand callback (weights are in tons)
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            # Handle potential NaN values
            demand = data['demands'][from_node]
            if demand is None or demand != demand:  # Check for NaN
                return 0
            # Convert tons to integer (multiply by 1000 to preserve 3 decimal places)
            return int(demand * 1000)

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Add capacity dimension with proper vehicle capacities
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,    # no slack
            # Convert tons to integer (multiply by 1000 to preserve 3 decimal places)
            [int(cap * 1000) for cap in data['vehicle_capacities']],
            True,  # start cumul to zero
            'Capacity')
        
        # Set first solution strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = strategy
        
        # Use Guided Local Search
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        
        # Set time limits
        search_parameters.time_limit.seconds = time_limit
        search_parameters.solution_limit = 100
        
        # Enable full propagation
        search_parameters.use_full_propagation = True
        
        # Set other parameters to help find feasible solutions
        search_parameters.guided_local_search_lambda_coefficient = 0.1
        search_parameters.use_depth_first_search = True
        search_parameters.use_cp = True
        search_parameters.use_cp_sat = True
        
        print(f"Solving with {data['num_vehicles']} vehicles...")
        print(f"Total demand: {sum(data['demands'])} tons")
        print(f"Total capacity: {sum(data['vehicle_capacities'])} tons")
        
        # Solve and get solution
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print("Found feasible solution!")
            return self._convert_or_tools_solution(solution, manager, routing, data)
        else:
            print("No feasible solution found with current parameters")
            return None
    
    def _convert_or_tools_solution(self, solution, manager, routing, data) -> List[Route]:
        """
        Convert OR-Tools solution to our Route objects.
        
        Args:
            solution: OR-Tools solution
            manager: RoutingIndexManager
            routing: RoutingModel
            data: Problem data
            
        Returns:
            List of Route objects
        """
        routes = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            if solution.Value(routing.NextVar(index)) == routing.End(vehicle_id):
                continue  # Skip empty routes
            
            # Safely get truck type
            truck_type_idx = vehicle_id % len(data['truck_types'])
            truck_type = data['truck_types'][truck_type_idx]
            
            # Initialize route with warehouse
            route = Route(
                vehicle_id=f"V_{truck_type}_{vehicle_id}", 
                locations=[self.warehouse_location], 
                parcels=[],
                data_processor=self.data_processor,
                vehicle_capacity=self.data_processor.truck_specifications[truck_type]['weight_capacity'] * 1000  # Convert tons to kg
            )
            
            # Add locations and parcels
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                city = data['cities'][node_idx]
                
                # Find orders for this destination
                for order_id, (source, dest) in data['order_locations'].items():
                    if dest == city:
                        source_loc = self._create_location(source)
                        dest_loc = self._create_location(dest)
                        
                        # Add locations
                        route.locations.extend([source_loc, dest_loc])
                        
                        # Get order details
                        order_mask = self.data_processor.order_data['Order_ID'] == order_id
                        if not any(order_mask):
                            continue
                            
                        # Convert weight from tons to kg
                        weight = float(self.data_processor.order_data[order_mask]['Weight'].values[0]) * 1000
                        int_order_id = self.data_processor.get_order_id_int(order_id)
                        
                        # Add parcel
                        route.parcels.append(Parcel(
                            id=int_order_id, 
                            destination=dest_loc,
                            source=source_loc,
                            weight=weight
                        ))
                
                index = solution.Value(routing.NextVar(index))
            
            # Add warehouse at end if needed
            if route.locations[-1] != self.warehouse_location:
                route.locations.append(self.warehouse_location)
            
            # Calculate distance and cost
            route.calculate_total_distance()
            route.total_cost = route.total_distance * self.data_processor.truck_specifications[truck_type]['cost_per_km']
            
            routes.append(route)
        
        return routes
    
    def _enhance_with_ml(self, routes: List[Route]) -> List[List[Route]]:
        """
        Use ML to enhance OR-Tools solutions.
        
        Args:
            routes: List of routes from OR-Tools
            
        Returns:
            List of enhanced solutions
        """
        print("Enhancing solutions with ML...")
        
        # Create 3 different enhanced solutions
        enhanced_solutions = []
        
        # If no routes to enhance, return empty solution array
        if not routes:
            print("Warning: No routes to enhance with ML")
            return [[]]  # Return array with one empty solution
        
        # Group routes by similar characteristics for solution formation
        route_groups = self._group_routes(routes)
        
        # Create solutions from different groupings
        for i in range(3):
            solution = []
            
            # Use a different grouping strategy for each solution
            if i == 0:
                # Solution 1: Truck type optimization
                solution = self._optimize_truck_types(routes)
            elif i == 1:
                # Solution 2: Load balancing
                solution = self._balance_loads(routes)
            else:
                # Solution 3: Distance minimization
                solution = self._minimize_distances(routes)
            
            if solution:
                enhanced_solutions.append(solution)
        
        # Add original OR-Tools routes as a backup solution
        if routes:
            # Group routes by vehicle ID to form a solution
            vehicle_routes = {}
            for route in routes:
                vehicle_id = route.vehicle_id
                if vehicle_id not in vehicle_routes:
                    vehicle_routes[vehicle_id] = route
            
            original_solution = list(vehicle_routes.values())
            if original_solution:
                enhanced_solutions.append(original_solution)
        
        # If no solutions were created, return the original routes as a single solution
        if not enhanced_solutions and routes:
            enhanced_solutions.append(routes)
        
        # If still no solutions, return array with one empty solution
        if not enhanced_solutions:
            enhanced_solutions.append([])
            
        return enhanced_solutions
    
    def _group_routes(self, routes: List[Route]) -> Dict[str, List[Route]]:
        """
        Group routes by characteristics for solution formation.
        
        Args:
            routes: List of routes
            
        Returns:
            Dictionary with grouped routes
        """
        groups = {
            'by_truck': {},  # Group by truck type
            'by_region': {},  # Group by geographic region
            'by_load': {}  # Group by load level
        }
        
        if not routes:
            return groups
            
        for route in routes:
            # Group by truck type
            truck_type = route.vehicle_id.split('_')[1]
            if truck_type not in groups['by_truck']:
                groups['by_truck'][truck_type] = []
            groups['by_truck'][truck_type].append(route)
            
            # Group by load level (high, medium, low)
            load_factor = route.get_total_weight() / route.vehicle_capacity
            load_category = 'high' if load_factor > 0.7 else 'medium' if load_factor > 0.4 else 'low'
            if load_category not in groups['by_load']:
                groups['by_load'][load_category] = []
            groups['by_load'][load_category].append(route)
            
            # Group by region (simplified - just use first parcel destination)
            if route.parcels:
                region = route.parcels[0].destination.city_name[:2]  # First two letters as region
                if region not in groups['by_region']:
                    groups['by_region'][region] = []
                groups['by_region'][region].append(route)
        
        return groups
    
    def _optimize_truck_types(self, routes: List[Route]) -> List[Route]:
        """
        Optimize truck types based on ML predictions.
        
        Args:
            routes: List of routes
            
        Returns:
            List of routes with optimized truck types
        """
        if not routes:
            return []
            
        optimized_routes = []
        
        for route in routes:
            # Skip if no parcels
            if not route.parcels:
                continue
                
            # Current truck details
            truck_type = route.vehicle_id.split('_')[1]
            current_capacity = self.data_processor.truck_specifications[truck_type]['weight_capacity']
            
            # Get total weight
            total_weight = route.get_total_weight()
            load_factor = total_weight / current_capacity
            
            # Check if truck type change would be beneficial
            if load_factor < 0.5:  # Underutilized
                # Try smaller truck
                truck_types = ['9.6', '12.5', '16.5']
                current_idx = truck_types.index(truck_type) if truck_type in truck_types else 0
                
                if current_idx > 0:  # Not already using smallest truck
                    smaller_type = truck_types[current_idx - 1]
                    smaller_capacity = self.data_processor.truck_specifications[smaller_type]['weight_capacity']
                    
                    # Check if smaller truck can handle the load
                    if total_weight <= smaller_capacity:
                        # Convert route data for ML prediction
                        route_data = {
                            'parcels': [{'destination': p.destination.city_name} for p in route.parcels],
                            'total_weight': total_weight,
                            'total_distance': route.total_distance,
                            'vehicle_capacity': smaller_capacity,
                            'truck_type': smaller_type
                        }
                        
                        # Predict new cost
                        new_cost = self.route_predictor.predict_route_cost(route_data)
                        
                        # If cost improved, use smaller truck
                        if new_cost < route.total_cost:
                            # Create new route with smaller truck
                            new_route = copy.deepcopy(route)
                            new_route.vehicle_id = f"V_{smaller_type}_{route.vehicle_id.split('_')[-1]}"
                            new_route.vehicle_capacity = smaller_capacity
                            new_route.total_cost = new_cost
                            optimized_routes.append(new_route)
                            continue
            
            # If no change or not beneficial, keep original route
            optimized_routes.append(route)
        
        return optimized_routes
    
    def _balance_loads(self, routes: List[Route]) -> List[Route]:
        """
        Balance loads across vehicles based on ML predictions.
        
        Args:
            routes: List of routes
            
        Returns:
            List of routes with balanced loads
        """
        if not routes:
            return []
            
        # Sort routes by load factor (ascending)
        sorted_routes = sorted(routes, key=lambda r: r.get_total_weight() / r.vehicle_capacity)
        
        # Identify very light and very heavy routes
        light_routes = [r for r in sorted_routes if r.get_total_weight() / r.vehicle_capacity < 0.3]
        heavy_routes = [r for r in sorted_routes if r.get_total_weight() / r.vehicle_capacity > 0.8]
        
        # If no imbalance, return original routes
        if not light_routes or not heavy_routes:
            return routes
        
        # Try to move parcels from heavy to light routes
        balanced_routes = []
        for route in routes:
            # Skip if already processed
            if route in balanced_routes:
                continue
            
            # If it's a heavy route, try to offload parcels
            if route in heavy_routes:
                # Find a light route
                for light_route in light_routes:
                    if light_route in balanced_routes:
                        continue
                    
                    # Try to move some parcels
                    moved = self._move_parcels(route, light_route)
                    if moved:
                        balanced_routes.extend(moved)
                        break
                else:
                    # No suitable light route found, keep original
                    balanced_routes.append(route)
            else:
                # Not a heavy route, keep as is for now
                balanced_routes.append(route)
        
        return balanced_routes
    
    def _move_parcels(self, from_route: Route, to_route: Route) -> Optional[List[Route]]:
        """
        Try to move parcels between routes to balance loads.
        
        Args:
            from_route: Source route (heavy)
            to_route: Destination route (light)
            
        Returns:
            List of modified routes or None if not possible
        """
        # Check if routes have parcels
        if not from_route.parcels or not to_route.parcels:
            return None
        
        # Calculate available capacity in light route
        to_capacity = to_route.vehicle_capacity
        to_weight = to_route.get_total_weight()
        available_capacity = to_capacity - to_weight
        
        # Find candidate parcels to move (up to 30% of from_route parcels)
        num_to_move = max(1, len(from_route.parcels) // 3)
        candidates = sorted(from_route.parcels, key=lambda p: p.weight)[:num_to_move]
        
        # Check if any combination can fit
        total_candidate_weight = sum(p.weight for p in candidates)
        if total_candidate_weight > available_capacity:
            return None
        
        # Create new routes with moved parcels
        new_from_route = copy.deepcopy(from_route)
        new_to_route = copy.deepcopy(to_route)
        
        # Remove candidates from from_route
        new_from_route.parcels = [p for p in new_from_route.parcels if p.id not in [c.id for c in candidates]]
        
        # Add candidates to to_route
        new_to_route.parcels.extend(candidates)
        
        # Recalculate routes
        # This is simplified - in a real system we'd need to recalculate the entire route
        new_from_route.calculate_total_distance()
        new_to_route.calculate_total_distance()
        
        # Update costs
        truck_type_from = new_from_route.vehicle_id.split('_')[1]
        truck_type_to = new_to_route.vehicle_id.split('_')[1]
        
        new_from_route.total_cost = new_from_route.total_distance * \
            self.data_processor.truck_specifications[truck_type_from]['cost_per_km']
        new_to_route.total_cost = new_to_route.total_distance * \
            self.data_processor.truck_specifications[truck_type_to]['cost_per_km']
        
        return [new_from_route, new_to_route]
    
    def _minimize_distances(self, routes: List[Route]) -> List[Route]:
        """
        Optimize routes to minimize distances based on ML predictions.
        
        Args:
            routes: List of routes
            
        Returns:
            List of routes with minimized distances
        """
        if not routes:
            return []
            
        optimized_routes = []
        
        for route in routes:
            # Skip if no parcels
            if not route.parcels:
                continue
            
            # Original route data
            original_data = {
                'parcels': [{'destination': p.destination.city_name} for p in route.parcels],
                'total_weight': route.get_total_weight(),
                'total_distance': route.total_distance,
                'vehicle_capacity': route.vehicle_capacity,
                'truck_type': route.vehicle_id.split('_')[1]
            }
            
            # Get ML prediction for original route
            predicted_cost = self.route_predictor.predict_route_cost(original_data)
            
            # If ML predicts significantly lower cost, there might be optimization potential
            if predicted_cost < route.total_cost * 0.85:  # >15% potential savings
                # Create a new route with optimized sequence
                new_route = copy.deepcopy(route)
                
                # Reorder locations - this is simplified as a real implementation
                # would use a more sophisticated reordering algorithm
                # Here we're just demonstrating the concept
                locations = []
                
                # Start with warehouse
                locations.append(route.locations[0])  # Warehouse
                
                # Add all source locations
                source_locs = []
                for parcel in route.parcels:
                    if parcel.source and parcel.source not in source_locs:
                        source_locs.append(parcel.source)
                
                # Sort source locations by distance from warehouse
                source_locs.sort(key=lambda loc: self._calculate_distance(
                    route.locations[0], loc))
                
                locations.extend(source_locs)
                
                # Add all destination locations
                dest_locs = []
                for parcel in route.parcels:
                    if parcel.destination and parcel.destination not in dest_locs:
                        dest_locs.append(parcel.destination)
                
                # Sort destination locations by distance from last source location
                if source_locs:
                    dest_locs.sort(key=lambda loc: self._calculate_distance(
                        source_locs[-1], loc))
                
                locations.extend(dest_locs)
                
                # End with warehouse
                locations.append(route.locations[0])  # Warehouse
                
                # Update route with new locations
                new_route.locations = locations
                
                # Recalculate distance and cost
                new_route.calculate_total_distance()
                truck_type = new_route.vehicle_id.split('_')[1]
                new_route.total_cost = new_route.total_distance * \
                    self.data_processor.truck_specifications[truck_type]['cost_per_km']
                
                optimized_routes.append(new_route)
            else:
                # No significant improvement potential, keep original
                optimized_routes.append(route)
        
        return optimized_routes
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate distance between two locations.
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance between locations
        """
        if self.data_processor:
            idx1 = self.data_processor.city_to_idx.get(loc1.city_name, 0)
            idx2 = self.data_processor.city_to_idx.get(loc2.city_name, 0)
            return self.data_processor.distance_matrix[idx1][idx2]
        else:
            # Fallback if distance matrix not available
            return ((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2) ** 0.5
    
    def _get_warehouse_location(self) -> Location:
        """
        Get warehouse location.
        
        Returns:
            Location object for warehouse
        """
        warehouse_city = list(self.data_processor.cities)[0]
        return self._create_location(warehouse_city)
    
    def _create_location(self, city_name: str) -> Location:
        """
        Create Location object from city name with caching.
        
        Args:
            city_name: Name of the city
            
        Returns:
            Location object
        """
        if city_name not in self._location_cache:
            idx = self.data_processor.city_to_idx.get(city_name, 0)
            self._location_cache[city_name] = Location(
                city_name=city_name,
                lat=float(idx),  # Use index as coordinate for visualization
                lon=float(idx)   # Use index as coordinate for visualization
            )
        return self._location_cache[city_name]

    def evaluate_solution(self, solution: List[Route]) -> Dict:
        """
        Evaluate a solution with detailed metrics.
        
        Args:
            solution: List of routes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not solution:
            return {
                'total_distance': float('inf'),
                'total_cost': float('inf'),
                'parcels_delivered': 0,
                'capacity_utilization': 0.0,
                'route_efficiency': 0.0
            }

        # Calculate basic metrics
        total_distance = sum(route.total_distance for route in solution)
        total_cost = sum(route.total_cost for route in solution)
        total_parcels = sum(len(route.parcels) for route in solution)
        
        # Calculate capacity utilization
        utilization_scores = []
        for route in solution:
            if route.vehicle_capacity > 0:
                utilization = route.get_total_weight() / route.vehicle_capacity
                utilization_scores.append(utilization)
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.0
        
        # Calculate route efficiency (parcels per km)
        total_parcels_weighted = sum(len(route.parcels) * route.get_total_weight() 
                                   for route in solution)
        route_efficiency = total_parcels_weighted / total_distance if total_distance > 0 else 0.0
        
        # Calculate percentage of total orders delivered
        total_orders = len(self.data_processor.order_data)
        delivery_ratio = total_parcels / total_orders if total_orders > 0 else 0.0
        
        return {
            'total_distance': total_distance,
            'total_cost': total_cost,
            'parcels_delivered': total_parcels,
            'capacity_utilization': avg_utilization,
            'route_efficiency': route_efficiency,
            'delivery_ratio': delivery_ratio
        }