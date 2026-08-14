#!/usr/bin/env python3
"""
Test script for Causal Memory Layer.

Demonstrates:
1. Creating events with causal edges
2. Querying causal paths
3. Handling contradictions/overwrites
4. Temporal reasoning
"""

import sys
import time
from causal_memory import CausalMemory


def main():
    print("=" * 60)
    print("Causal Memory Layer - Test Script")
    print("=" * 60)
    
    # Initialize memory
    memory = CausalMemory(
        url="http://localhost:8443",
        auth_token="local-dev-auth-token-32-characters-long",
        admin_url="http://localhost:9090"
    )
    
    # Check connection
    print("\n1. Checking HydraDB connection...")
    if memory.client.health_check() or memory.client.ready_check():
        print("   ✓ HydraDB is healthy")
    else:
        print("   ✗ Cannot connect to HydraDB")
        print("   Make sure HydraDB is running on localhost:8443")
        sys.exit(1)
    
    # Create events
    print("\n2. Creating events...")
    
    e1 = memory.add_event(
        text="User moved to San Francisco",
        session_id="session-1",
        topic="location",
        timestamp=int(time.time()) - 10000
    )
    print(f"   Created: {e1.id} - {e1.text}")
    
    e2 = memory.add_event(
        text="User found apartment in Mission district",
        session_id="session-2",
        topic="housing",
        timestamp=int(time.time()) - 8000
    )
    print(f"   Created: {e2.id} - {e2.text}")
    
    e3 = memory.add_event(
        text="User loves the neighborhood",
        session_id="session-3",
        topic="satisfaction",
        timestamp=int(time.time()) - 5000
    )
    print(f"   Created: {e3.id} - {e3.text}")
    
    e4 = memory.add_event(
        text="User prefers dark mode",
        session_id="session-4",
        topic="preference",
        timestamp=int(time.time()) - 3000
    )
    print(f"   Created: {e4.id} - {e4.text}")
    
    e5 = memory.add_event(
        text="User switched to dark mode because of eye strain",
        session_id="session-5",
        topic="preference_reason",
        timestamp=int(time.time()) - 2000
    )
    print(f"   Created: {e5.id} - {e5.text}")
    
    # Create causal relationships
    print("\n3. Creating causal relationships...")
    
    r1 = memory.add_causal_relation(
        source_id=e1.id,
        target_id=e2.id,
        relation_type="CAUSES",
        confidence=0.9,
        mechanism="relocation led to housing search"
    )
    print(f"   {e1.id} → {e2.id} (CAUSES)")
    
    r2 = memory.add_causal_relation(
        source_id=e2.id,
        target_id=e3.id,
        relation_type="CAUSES",
        confidence=0.85,
        mechanism="found apartment led to satisfaction"
    )
    print(f"   {e2.id} → {e3.id} (CAUSES)")
    
    r3 = memory.add_causal_relation(
        source_id=e5.id,
        target_id=e4.id,
        relation_type="CAUSES",
        confidence=0.95,
        mechanism="eye strain caused preference change"
    )
    print(f"   {e5.id} → {e4.id} (CAUSES)")
    
    # Query causal paths
    print("\n4. Querying causal paths...")
    
    print(f"\n   Query: Causal path from {e1.id} to {e3.id}")
    paths = memory.find_causal_path(e1.id, e3.id, max_hops=5)
    print(f"   Found {len(paths)} path(s)")
    
    print(f"\n   Query: All effects of {e1.id}")
    effects = memory.find_all_effects(e1.id, max_hops=3)
    print(f"   Found {len(effects)} effect(s)")
    
    # Search
    print("\n5. Searching for events...")
    
    results = memory.search("San Francisco")
    print(f"   Search 'San Francisco': {len(results)} result(s)")
    for r in results:
        print(f"     - {r.text}")

    results = memory.search("dark mode")
    print(f"   Search 'dark mode': {len(results)} result(s)")
    for r in results:
        print(f"     - {r.text}")
    
    # Test current fact retrieval
    print("\n6. Testing current fact retrieval...")
    
    current = memory.find_current_fact("location")
    if current:
        print(f"   Current location fact: {current.text}")
    else:
        print("   No current location fact found")
    
    # Test overwrite scenario
    print("\n7. Testing overwrite scenario...")
    
    e6 = memory.add_event(
        text="User moved to New York",
        session_id="session-6",
        topic="location",
        timestamp=int(time.time())
    )
    print(f"   Created: {e6.id} - {e6.text}")
    
    memory.add_overwrite(
        old_event_id=e1.id,
        new_event_id=e6.id,
        reason="user moved again"
    )
    print(f"   {e6.id} OVERWRITES {e1.id}")
    
    current = memory.find_current_fact("location")
    if current:
        print(f"   Current location fact: {current.text}")
        if current.id == e6.id:
            print("   ✓ Overwrite correctly handled")
        else:
            print("   ✗ Overwrite not correctly handled")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
