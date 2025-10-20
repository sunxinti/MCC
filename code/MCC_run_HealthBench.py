#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author. Xinti Sun

"""
HealthBench evaluation script using MCC framework

Usage:
1. Run complete HealthBench evaluation:
   python MCC_run_HealthBench.py.py

2. Run HealthBench Hard subset:
   python MCC_run_HealthBench.py --subset hard

3. Run HealthBench Consensus subset:
   python MCC_run_HealthBench.py --subset consensus

4. Specify samples:
   python MCC_run_HealthBench.py --subset consensus --sample_indices "1001-2000"
"""

# ===================== Configuration Section =====================
import argparse
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MCC_Henalthbench_function import run_healthbench_evaluation, MCCSampler

def main():
    parser = argparse.ArgumentParser(
        description='Run HealthBench evaluation using MCC framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--subset', 
        type=str, 
        choices=['hard', 'consensus'], 
        help='HealthBench subset (hard, consensus, or leave empty for full dataset)'
    )
    
    parser.add_argument(
        '--examples', 
        type=int, 
        help='Limit number of evaluation samples (default: use all samples)'
    )
    
    parser.add_argument(
        '--max_rounds', 
        type=int, 
        default=3, 
        help='Maximum debate rounds for MCC framework (default: 3)'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Run simple test, process only 1 sample'
    )
    
    parser.add_argument(
        '--sample_indices',
        type=str,
        default=None,
        help='Specify sample indices to evaluate, e.g.: "0,5,10" or "0-5" or single number "3"'
    )
    
    args = parser.parse_args()
    
    # Parse sample indices
    sample_indices = None
    if args.sample_indices:
        try:
            if '-' in args.sample_indices:
                # Handle range format "0-5"
                start, end = map(int, args.sample_indices.split('-'))
                sample_indices = list(range(start, end + 1))
            elif ',' in args.sample_indices:
                # Handle list format "0,5,10"
                sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
            else:
                # Handle single number "3"
                sample_indices = [int(args.sample_indices)]
            print(f"Specified sample indices: {sample_indices}")
        except ValueError:
            print(f"Error: Invalid sample indices format: {args.sample_indices}")
            print("Supported formats: '3' or '0,5,10' or '0-5'")
            return
    
    # If test mode, limit sample count
    if args.test:
        args.examples = 1
        print("Test mode: processing only 1 sample")
    
    print("="*80)
    print("HealthBench Evaluation - Using MCC Framework")
    print("="*80)
    print(f"Subset: {args.subset or 'Full dataset'}")
    if sample_indices:
        print(f"Specified samples: {sample_indices}")
    else:
        print(f"Sample count: {args.examples or 'All'}")
    print(f"Maximum debate rounds: {args.max_rounds}")
    print("="*80)
    
    try:
        # Run HealthBench evaluation
        results = run_healthbench_evaluation(
            subset_name=args.subset,
            num_examples=args.examples,
            max_rounds=args.max_rounds,
            sample_indices=sample_indices
        )
        
        print(f"\nEvaluation completed! Processed {len(results)} samples")
        
        # Calculate success rate
        successful_results = [r for r in results if 'error' not in r]
        success_rate = len(successful_results) / len(results) if results else 0
        
        print(f"Success rate: {success_rate:.2%}")
        
        if args.test and successful_results:
            print("\nMCC response for test sample:")
            print("-" * 60)
            print(successful_results[0]['mcc_response'][:500] + "..." if len(successful_results[0]['mcc_response']) > 500 else successful_results[0]['mcc_response'])
            print("-" * 60)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
