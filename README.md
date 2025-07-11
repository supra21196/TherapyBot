This is a passion project of mine that combines two domains: data science and mental health advocacy. This project applies my data science knowledge towards creating a personalized therapeutic support system that helps people access coping strategies and breathing techniques when they need them most.

TherapyBot uses local AI processing to understand your emotional state and retrieves specific, evidence-based therapeutic techniques - not just generic advice. It learns from user feedback to personalize responses while maintaining complete privacy through local data storage.

Built for anyone who's ever felt overwhelmed and just needed to remember how to breathe, this project represents my commitment to making mental health resources freely accessible through thoughtful application of data science principles.

## Project Overview:
### Data Architecture:
The system uses curated evidence-based therapeutic techniques stored locally with sentence transformers (all-MiniLM-L6-v2) for 384-dimensional semantic embeddings. I manually curated the knowledge base rather than web scraping to ensure safety and quality. The intelligent routing system integrates external APIs for current research (PubMed), condition facts, and local resources while keeping personal support queries local.
### Performance Tracking:
Each interaction generates confidence scores through cosine similarity search, processing time metrics, and user feedback ratings (1-5 stars). The SQLite database captures query patterns, source performance analysis, and effectiveness correlations while maintaining complete local privacy - no external data sharing.

## What This Actually Does

Your personal mental health toolkit that remembers everything for you

Ask it things like:
- "I'm having a panic attack right now"
- "I can't get out of bed today"  
- "My mind is racing and I can't sleep"
- "What's that breathing technique for anxiety?"

And it'll give you specific, helpful techniques from actual therapeutic practices - not just generic "breathe deeply" advice.

## Why I Built This

Let's be honest - mental health apps are either too simplistic ("just meditate!") or too overwhelming (17 different mood tracking features). I wanted something that works like having a really smart, caring friend who happens to know a lot about therapy techniques.

It's also completely private. No accounts, no data sharing, no "we noticed you've been anxious lately" notifications. Just you and your personal knowledge base.

## The Smart Part

Here's what makes it actually useful instead of just another chatbot:

**It knows the difference between different types of questions:**
- Personal support ("I feel terrible") → Uses your therapeutic knowledge base
- Research questions ("What's the latest on depression treatment?") → Tries to find current information
- Crisis situations ("I'm thinking of hurting myself") → Immediately provides emergency resources

**It learns from your feedback:**
- Rate responses 1-5 stars
- The system learns which techniques work best for you
- Builds a personalized understanding over time

**It's genuinely helpful in emergencies:**
- Recognizes crisis language
- Provides immediate coping techniques
- Always includes real crisis hotline numbers (988, 911)

## How to Use It

### Quick Start
```bash
pip install sentence-transformers aiosqlite numpy
python main.py
```

Choose option 2 for a demo, or option 1 to start asking your own questions.

### Your First Conversation
```
You: I'm feeling really anxious about a presentation tomorrow

Assistant: Here's a technique that can help:

5-4-3-2-1 Grounding for Anxiety: Notice 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, 1 thing you can taste. This brings you back to the present moment when feeling overwhelmed.

### Adding Your Own Techniques
The system comes with evidence-based techniques, but you can add your own:

```python
await system.add_knowledge(
    "When I feel overwhelmed at work, I step outside for 2 minutes and focus on tree sounds",
    {"category": "work_stress", "personal": True}
)
```

## What's Actually In Here

**Real therapeutic techniques:**
- 5-4-3-2-1 grounding for anxiety
- Box breathing for panic attacks
- Progressive muscle relaxation
- RAIN technique for difficult emotions
- Gentle morning routines for depression
- Sleep techniques for racing minds
- Crisis survival strategies (TIPP technique)

**Smart features:**
- Recognizes 25+ mental health keywords
- Routes urgent queries appropriately
- Tracks what helps you most
- Provides crisis resources when needed

## The Technical Stuff (For Nerds)

- **No cloud AI**: Uses sentence-transformers for local text understanding
- **No API costs**: Everything runs on your machine
- **Smart routing**: Knows when to search internal knowledge vs external sources
- **Privacy focused**: SQLite database stored locally
- **Lightweight**: Works in CodeSandbox and low-resource environments

It's basically a RAG (Retrieval-Augmented Generation) system, but instead of generating new text, it retrieves proven therapeutic techniques. More reliable, less AI hallucination.

## When NOT to Use This

**This is not a replacement for:**
- Professional therapy
- Crisis counseling
- Medical advice
- Medication management

**If you're having thoughts of self-harm:** Please call 988 (Crisis Lifeline) or 911 immediately. This tool includes those resources, but human support is what you need.

## Real Talk About Mental Health Tech

Most mental health apps are built by people who've never had a panic attack at 3 AM. This was built by someone who has, and who got tired of scrolling through meditation apps when what I actually needed was that specific grounding technique my therapist taught me.

It's not trying to be your therapist. It's trying to be the friend who remembers all the useful stuff when your brain is too fried to recall it.

## Contributing Your Own Techniques

Got a coping strategy that works for you? Add it:

```python
await system.add_knowledge(
    "Your technique here...",
    {"category": "anxiety", "source": "personal", "effectiveness": "high"}
)
```

The more real techniques we collect, the more helpful this becomes for everyone.

## Support & Community

*Built with care for anyone who's ever felt overwhelmed and just needed to remember how to breathe

## License

MIT - Use it, modify it, share it. Mental health resources should be freely available to everyone.
