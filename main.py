"""
Mental Health RAG Assistant - Main Entry Point
Simple interface to run your personal mental health knowledge base
"""

import asyncio
import os
from datetime import datetime

# Simple configuration - no config file needed
CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "db_path": "mental_health_rag.db",
    "debug": True,
    "vector_dim": 384,
    "max_documents": 50,  # Limit for CodeSandbox
    "query_timeout": 10   # Seconds
}

async def setup_mental_health_knowledge():
    """Setup the mental health knowledge base with therapeutic techniques."""
    from rag_agent import MentalHealthRAG
    
    print("🧠 Setting up your Mental Health Assistant...")
    
    # Initialize system
    system = MentalHealthRAG()
    
    # Mental health techniques - real therapeutic content
    techniques = [
        {
            "content": "5-4-3-2-1 Grounding for Anxiety: Notice 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, 1 thing you can taste. This brings you back to the present moment when feeling overwhelmed or having a panic attack.",
            "category": "anxiety", "urgency": "immediate", "duration": "2_min"
        },
        {
            "content": "Box Breathing for Panic: Inhale for 4 counts, hold for 4, exhale for 4, hold empty for 4. Repeat 4-6 times. This activates your parasympathetic nervous system and slows your heart rate. Use when you feel panic rising.",
            "category": "anxiety", "urgency": "immediate", "duration": "2_min"
        },
        {
            "content": "Gentle Morning Routine for Depression: 1) Open curtains immediately when you wake up. 2) Drink a full glass of water. 3) Do 10 gentle stretches in bed. 4) Write one tiny thing you're grateful for. 5) Set one micro-goal like 'brush teeth'. No pressure, just gentle momentum.",
            "category": "depression", "time": "morning", "difficulty": "low"
        },
        {
            "content": "The 2-Minute Rule for Depression: When everything feels impossible, commit to just 2 minutes. 2 minutes of cleaning, walking, journaling, or calling a friend. Often the hardest part is starting. You can stop after 2 minutes or keep going if you feel like it.",
            "category": "depression", "difficulty": "low", "duration": "2_min"
        },
        {
            "content": "Progressive Muscle Relaxation: Start with your toes - tense for 5 seconds, then release. Move up through calves, thighs, abdomen, hands, arms, shoulders, face. Notice the contrast between tension and relaxation. Great for bedtime anxiety.",
            "category": "anxiety", "time": "evening", "duration": "10_min"
        },
        {
            "content": "Stress Reset Protocol: Stop what you're doing. Take 3 deep breaths. Ask yourself: 'Is this urgent or just feels urgent?' If not truly urgent, step away for 10 minutes. Go outside, stretch, or listen to one song. Return with fresh perspective.",
            "category": "stress", "urgency": "immediate", "duration": "10_min"
        },
        {
            "content": "RAIN Technique for Difficult Emotions: Recognize what you're feeling. Allow the emotion to be there without fighting it. Investigate with kindness - where do you feel it in your body? Non-attachment - remind yourself this feeling will pass.",
            "category": "mindfulness", "technique": "RAIN", "use_case": "emotional_regulation"
        },
        {
            "content": "Thought Record for Negative Spirals: Write down the negative thought. Rate how much you believe it (1-10). List evidence for and against it. Write a more balanced thought. Rate belief in the balanced thought. This helps break cycles of catastrophic thinking.",
            "category": "cognitive", "technique": "thought_record", "condition": "negative_thinking"
        },
        {
            "content": "Crisis Survival Kit: When in emotional crisis, use TIPP - Temperature (cold water on face), Intense exercise (jumping jacks for 1 minute), Paced breathing (long exhales), Paired muscle relaxation. These quickly change your body chemistry.",
            "category": "crisis", "urgency": "emergency", "technique": "TIPP"
        },
        {
            "content": "Opposite Action for Depression: When depression tells you to isolate, reach out to someone. When it says stay in bed, get up and move your body for 5 minutes. When it says you're worthless, do one kind thing for yourself. Act opposite to what depression wants.",
            "category": "depression", "technique": "opposite_action"
        },
        {
            "content": "Racing Mind Bedtime Technique: Keep a notepad by your bed. When worries come up, write them down and tell yourself 'I'll deal with this tomorrow.' Do a body scan from toes to head, releasing tension. If still awake after 20 minutes, get up and do a quiet activity until sleepy.",
            "category": "sleep", "condition": "insomnia", "time": "bedtime"
        },
        {
            "content": "The Best Friend Test: When being self-critical, ask 'What would I tell my best friend if they were in this situation?' We're often much kinder to others than ourselves. Use that same compassionate voice for yourself.",
            "category": "cognitive", "technique": "self_compassion", "condition": "self_criticism"
        }
    ]
    
    # Add all techniques to knowledge base
    for i, technique in enumerate(techniques):
        success = await system.add_knowledge(
            technique["content"], 
            {k: v for k, v in technique.items() if k != "content"}
        )
        if success:
            print(f"✓ Added technique {i+1}/{len(techniques)}")
        else:
            print(f"✗ Failed to add technique {i+1}")
    
    print(f"\n🎉 Knowledge base ready with {len(techniques)} therapeutic techniques!")
    return system

async def interactive_session():
    """Interactive mental health assistant session."""
    
    # Setup knowledge base
    system = await setup_mental_health_knowledge()
    
    print("\n💚 Your Smart Mental Health Assistant is ready!")
    print("I can help with:")
    print("  🧠 Personal coping strategies (from my knowledge base)")
    print("  📚 Current mental health research and facts")
    print("  🏥 General information about conditions and treatments")
    print("  🆘 Crisis support and emergency resources")
    print("\nExamples:")
    print("  • 'I'm feeling anxious right now' (uses internal knowledge)")
    print("  • 'What is the latest research on depression?' (tries external sources)")
    print("  • 'What are the symptoms of ADHD?' (factual information)")
    print("\nType 'quit' to exit, 'help' for more examples")
    print("-" * 60)
    
    session_queries = []
    
    while True:
        try:
            user_input = input("\n💭 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            elif user_input.lower() == 'help':
                print("\n🆘 Quick commands:")
                print("  'emergency' - Crisis support techniques")
                print("  'anxiety' - Immediate anxiety relief")
                print("  'depression' - Depression support") 
                print("  'sleep' - Sleep and bedtime help")
                print("  'research depression' - Latest research on depression")
                print("  'what is anxiety' - Factual information about anxiety")
                continue
            elif user_input.lower() == 'emergency':
                user_input = "I'm in crisis and need immediate help"
            elif user_input.lower() == 'anxiety':
                user_input = "I'm feeling anxious and need help right now"
            elif user_input.lower() == 'depression':
                user_input = "I'm feeling depressed and have no motivation"
            elif user_input.lower() == 'sleep':
                user_input = "I can't sleep and my mind is racing"
            elif user_input.lower() == 'research depression':
                user_input = "What is the latest research on depression treatment?"
            elif user_input.lower() == 'what is anxiety':
                user_input = "What is anxiety disorder and what are the symptoms?"
            elif not user_input:
                print("Please enter a question or 'help' for examples.")
                continue
            
            # Get response from smart system
            print(f"🤗 Assistant: ", end="", flush=True)
            response = await system.query(user_input)
            print(response)
            
            # Store query
            session_queries.append(user_input)
            
            # Ask for rating
            rating_input = input("\nRate helpfulness 1-5 (Enter to skip): ").strip()
            if rating_input.isdigit() and 1 <= int(rating_input) <= 5:
                await system.add_feedback(user_input, float(rating_input))
                print(f"⭐ Thank you! Rated {rating_input}/5")
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")
    
    # Session summary with smart routing stats
    if session_queries:
        print(f"\n📋 This session you asked about:")
        for i, query in enumerate(session_queries[-5:], 1):  # Show last 5
            print(f"  {i}. {query[:50]}...")
        
        # Show routing statistics
        stats = await system.get_stats()
        print(f"\n📊 Smart Routing Stats:")
        print(f"  Internal knowledge used: {stats.get('internal_kb_calls', 0)} times")
        print(f"  External sources used: {stats.get('external_api_calls', 0)} times")
    
    await system.close()
    print("\n💙 Take care of yourself! Your mental health matters.")

async def demo_scenarios():
    """Run demonstration scenarios showing smart routing capabilities."""
    
    system = await setup_mental_health_knowledge()
    
    print("\n🧪 Demo: Smart Mental Health Assistant Routing\n")
    
    scenarios = [
        {
            "situation": "😰 Personal Anxiety (Internal Knowledge)",
            "query": "I'm having a panic attack right now and can't breathe",
            "expected_source": "internal"
        },
        {
            "situation": "📚 Research Question (External Source)",
            "query": "What is the latest research on depression treatment?",
            "expected_source": "external"
        },
        {
            "situation": "📊 Factual Information (External Source)",
            "query": "What are the statistics on anxiety disorders in adults?",
            "expected_source": "external"
        },
        {
            "situation": "😔 Personal Support (Internal Knowledge)", 
            "query": "I woke up feeling hopeless and can't get out of bed",
            "expected_source": "internal"
        },
        {
            "situation": "💊 Medical Information (External with Disclaimer)",
            "query": "What are the side effects of antidepressants?",
            "expected_source": "external"
        },
        {
            "situation": "🆘 Crisis (Internal Knowledge)",
            "query": "I'm thinking about ending my life",
            "expected_source": "internal"
        }
    ]
    
    for scenario in scenarios:
        print(f"{scenario['situation']}")
        print(f"💬 Query: '{scenario['query']}'")
        print(f"🎯 Expected routing: {scenario['expected_source']}")
        
        response = await system.query(scenario['query'])
        print(f"🤗 Response: {response[:150]}...")
        
        print("=" * 80)
        
        # Simulate positive feedback
        await system.add_feedback(scenario['query'], 4.5)
    
    # Show comprehensive stats
    stats = await system.get_stats()
    print(f"\n📊 Smart System Stats:")
    print(f"  Total queries: {stats.get('total_queries', 0)}")
    print(f"  Average rating: {stats.get('avg_rating', 0):.1f}/5.0")
    print(f"  Knowledge base size: {stats.get('total_documents', 0)} techniques")
    print(f"  Internal routing: {stats.get('internal_kb_calls', 0)} queries")
    print(f"  External routing: {stats.get('external_api_calls', 0)} queries")
    print(f"  System intelligence: ✅ Smart routing active")
    
    await system.close()

def main():
    """Main entry point with menu selection."""
    
    print("🌟 Mental Health RAG Assistant")
    print("Your personal therapeutic knowledge base")
    print("\nChoose your mode:")
    print("1. Interactive session (ask your own questions)")
    print("2. Demo scenarios (see system in action)")
    print("3. Setup only (just build knowledge base)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(interactive_session())
        elif choice == "2":
            asyncio.run(demo_scenarios())
        elif choice == "3":
            async def setup_only():
                system = await setup_mental_health_knowledge()
                await system.close()
                print("✅ Knowledge base setup complete!")
            asyncio.run(setup_only())
        else:
            print("Invalid choice. Running interactive session...")
            asyncio.run(interactive_session())
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Remember to be kind to yourself.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()