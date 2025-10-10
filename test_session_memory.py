#!/usr/bin/env python3
"""
Test script to verify session memory/context is working correctly.
"""
import requests
import json
import time

API_URL = "http://localhost:8000"

def test_session_memory():
    """Test that the AI remembers context across multiple queries."""

    print("Testing Session Memory for AI Chat Interface")
    print("=" * 50)

    # First query - analyze a specific filing
    query1 = "Analyze the SEC filing with accession number 0001326801-24-000090. What company is this for?"
    print(f"\n1. First query: {query1}")

    response1 = requests.post(
        f"{API_URL}/api/chat",
        json={
            "query": query1,
            "filters": {"accession_number": "0001326801-24-000090"}
        }
    )

    if response1.status_code != 200:
        print(f"Error: {response1.status_code}")
        print(response1.text)
        return

    data1 = response1.json()
    session_id = data1["session_id"]

    print(f"   Session ID: {session_id}")
    print(f"   Response preview: {data1['response'][:200]}...")

    # Wait a moment
    time.sleep(2)

    # Second query - should remember the filing context
    query2 = "What was the revenue mentioned in this filing?"
    print(f"\n2. Second query (using same session): {query2}")

    response2 = requests.post(
        f"{API_URL}/api/chat",
        json={
            "query": query2,
            "session_id": session_id  # Use the same session
        }
    )

    if response2.status_code != 200:
        print(f"Error: {response2.status_code}")
        print(response2.text)
        return

    data2 = response2.json()

    print(f"   Session ID: {data2['session_id']}")
    print(f"   Response preview: {data2['response'][:300]}...")

    # Check if the response shows context awareness
    if "0001326801-24-000090" in data2["response"] or "meta" in data2["response"].lower() or "facebook" in data2["response"].lower():
        print("\n✅ SUCCESS: AI remembered the filing context!")
    else:
        print("\n⚠️  WARNING: AI may not have remembered the filing context")

    # Third query - different topic but same filing
    query3 = "What risk factors were mentioned?"
    print(f"\n3. Third query (testing continued memory): {query3}")

    response3 = requests.post(
        f"{API_URL}/api/chat",
        json={
            "query": query3,
            "session_id": session_id
        }
    )

    if response3.status_code != 200:
        print(f"Error: {response3.status_code}")
        print(response3.text)
        return

    data3 = response3.json()
    print(f"   Response preview: {data3['response'][:300]}...")

    # Test with a new session to verify isolation
    print("\n4. Testing session isolation with new session...")
    query4 = "What filing were we discussing?"

    response4 = requests.post(
        f"{API_URL}/api/chat",
        json={
            "query": query4
            # No session_id - should create new session
        }
    )

    if response4.status_code != 200:
        print(f"Error: {response4.status_code}")
        print(response4.text)
        return

    data4 = response4.json()
    new_session_id = data4["session_id"]

    print(f"   New Session ID: {new_session_id}")
    print(f"   Response preview: {data4['response'][:300]}...")

    if new_session_id != session_id:
        print("\n✅ SUCCESS: New session created properly")
    else:
        print("\n❌ FAIL: Same session ID returned")

    # Check health endpoint for session stats
    print("\n5. Checking session manager stats...")
    health = requests.get(f"{API_URL}/health")
    if health.status_code == 200:
        health_data = health.json()
        if "services" in health_data and "session_manager" in health_data["services"]:
            stats = health_data["services"]["session_manager"]
            print(f"   Total sessions: {stats.get('total_sessions', 'N/A')}")
            print(f"   Max sessions: {stats.get('max_sessions', 'N/A')}")
            print(f"   TTL hours: {stats.get('session_ttl_hours', 'N/A')}")

    print("\n" + "=" * 50)
    print("Session Memory Test Complete!")

if __name__ == "__main__":
    try:
        test_session_memory()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()