# Reflection Log: Multi-Agent Travel Planner

## What I Learned from Implementing a Multi-Agent Workflow

Implementing this multi-agent travel planning system taught me the power of separation of concerns in AI applications. By dividing responsibilities between a Planner Agent (creative ideation) and a Reviewer Agent (factual validation), I created a workflow that mirrors real-world collaborative processes. The key insight was that each agent could specialize: the Planner leverages its broad knowledge to generate comprehensive itineraries without internet access, while the Reviewer uses the `internet_search` tool to verify facts and catch errors.

I learned that effective agent orchestration requires clear handoffs—the Planner's structured output becomes the Reviewer's input, and the Delta List mechanism provides transparent change tracking. This asynchronous collaboration pattern is scalable: the agents don't need to "talk" directly; they communicate through well-structured data formats (Markdown itineraries and explicit correction lists).

## Challenges Faced and How I Addressed Them

The primary challenge was crafting system prompts that produced reliable, consistent outputs. Initially, the Planner generated vague itineraries without specific times or costs. I addressed this by adding explicit formatting instructions and example structures in the prompt. For the Reviewer, I had to balance thoroughness with efficiency—encouraging targeted searches rather than exhaustive verification of every detail.

Another challenge was tool integration. Ensuring the Reviewer actually used `internet_search` required explicit instructions about when and how to search (e.g., "search for specific, verifiable facts like 'Louvre Museum closing days 2024'"). I also discovered that the Reviewer needed guidance on output format: presenting both the Delta List and the revised itinerary ensures transparency while delivering actionable results.

## Creative Ideas and Design Choices

I designed the Planner with a "budget consciousness" directive, asking it to track cumulative expenses and leave a 10-15% buffer. This makes itineraries more realistic. For the Reviewer, I introduced the "Delta List" concept—a structured changelog inspired by version control systems—to make validation transparent and educational for users.

I also emphasized role-appropriate constraints: the Planner works "offline" from existing knowledge, while the Reviewer is explicitly empowered with internet access. This division prevents the Planner from hallucinating facts while allowing the Reviewer to catch those hallucinations.

One persona decision was framing the Reviewer as "meticulous" but "constructive"—it should improve plans, not just criticize them. This tone engineering encourages helpful feedback rather than purely negative critique.

## External Tools and GenAI Assistance Used

I used Claude Code (Anthropic's AI assistant) to help structure the system prompts and debug the tool integration. Specifically, Claude helped me identify that the `tools` parameter needed to be a list containing the `internet_search` function object, not just a string reference.

I also referenced the Tavily API documentation to understand search result formatting and used GitHub Copilot for minor syntax suggestions in the markdown formatting examples within the prompts.

---

*Total word count: ~390 words (excluding this note and the external tools section title)*
