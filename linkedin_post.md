# LinkedIn Post Draft

---

**When should your AI agent ping you?**

Most AI agents send notifications on fixed schedules — "follow up in 2 hours," "remind tomorrow at 9 AM." But your availability isn't fixed. You're a morning person on weekdays and a night owl on weekends. You're responsive at 2 PM on Wednesdays but never at 4 PM on Fridays.

I built an adaptive scheduling system for ProactiveClaw (my proactive AI agent) that *learns* when you actually respond.

The approach: a **multi-armed bandit** with 168 arms — one for every hour of the week.

How it works:
- You respond to a notification within 10 min? That time slot gets reward 1.0
- You start a conversation on your own? That slot gets reward 0.5
- Haven't chatted at a time slot in a while? Its score decays toward zero

The update rule is dead simple:
```
Q[arm] = decay * Q_old + (1 - decay) * reward
```

The key insight: **the bandit doesn't schedule notifications directly.** It generates ranked recommendations that feed into the LLM's prompt. The LLM still makes the final call — it can override for urgent tasks, respect calendar conflicts, or spread out casual reminders.

This hybrid design gets you:
- Statistical learning from implicit feedback (no surveys, no setup)
- Contextual reasoning from the LLM (urgency, calendar, task type)
- Graceful cold start (no data = no recommendations = same behavior as before)
- Adaptation to changing routines via exponential time decay

The entire implementation is ~120 lines of Python. No ML frameworks. No database. Just a JSON file on disk.

Sometimes the best ML system is the simplest one that actually ships.

---

#AI #MachineLearning #MultiArmedBandit #AIAgents #ReinforcementLearning #ProductEngineering

---
