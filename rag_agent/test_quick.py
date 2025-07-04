import asyncio
from rag_agent import MentalHealthRAG

async def main():
    print("🧪 Quick CodeSandbox Test")
    
    system = MentalHealthRAG()
    
    # Add one technique
    success = await system.add_knowledge(
        "Take 3 deep breaths when feeling stressed",
        {"category": "stress"}
    )
    print(f"✅ Added knowledge: {success}")
    
    # Test query
    response = await system.query("I'm stressed")
    print(f"🤗 Response: {response[:100]}...")
    
    # Get stats
    stats = await system.get_stats()
    print(f"📊 Stats: {stats['total_documents']} techniques loaded")
    
    await system.close()
    print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(main())