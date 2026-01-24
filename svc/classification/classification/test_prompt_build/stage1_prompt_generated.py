"""
Stage 1: Category Detection Prompt

Auto-generated from condensed taxonomy and curated examples.
"""


def stage1_category_detection_prompt(comment: str) -> str:
    """
    Generate Stage 1 classification prompt for category detection.
    
    Args:
        comment: The conference feedback comment to analyze
        
    Returns:
        Formatted prompt string ready for LLM processing
    """
    return f"""You are an expert conference feedback analyzer. Your task is to identify which categories of feedback are present in a conference attendee comment.

COMMENT TO ANALYZE:
{comment}

---

CATEGORIES AND THEIR SCOPE:

**Attendee Engagement & Interaction**
Feedback about attendee connections, community building, and social aspects
- Community: Sense of belonging, community spirit, feeling welcomed, icebreaker activities
- Knowledge Exchange: Sharing experiences, collaborative learning, panel discussions, Q&A sessions
- Networking: Meeting new people, professional connections, peer discussions, speed networking
- Social Events: Gala dinners, receptions, informal gatherings, team-building activities

**Event Logistics & Infrastructure**
Feedback about event facilities, logistics, and infrastructure
- Conference Application/Software: Feedback on event apps, platforms, and digital tools
- Conference Venue: Attendee impressions of rooms, facilities, and accessibility
- Food/Beverages: Feedback on meals, snacks, drinks, and catering quality
- Hotel: General remarks about lodging arrangements and experience
- Technological Tools: Feedback on event tech, AV equipment, and digital services
- Transportation: Comments on travel logistics, shuttle services, and overall transport
- Wi-Fi: Feedback on Wi-Fi connectivity, speed, and overall satisfaction

**Event Operations & Management**
Feedback about overall event planning, coordination, and execution
- Conference: Broad remarks on event management, organization, or overall experience
- Conference Registration: Process of signing up, ease of registration, and any issues encountered
- Conference Scheduling: Timing of sessions, breaks, and program flow, including any conflicts or gaps
- Messaging & Awareness: Communication efforts, information dissemination, and clarity of updates

**Learning & Content Delivery**
Feedback about educational aspects, content quality, and delivery.
- Demonstration: Live demos, product showcases, hands-on examples, tech displays
- Gained Knowledge: New insights, skills acquired, understanding improved
- Open Discussion: Q&A sessions, audience participation, interactive discussions
- Panel Discussions: Expert panels, speaker debates, group presentations
- Presentations: Keynotes, speaker sessions, educational talks
- Resources/Materials: Handouts, slides, notes, books, online resources
- Session/Workshop: Breakout sessions, hands-on activities, skill-building workshops
- Topics: Subject matter, themes, content covered, session topics

**People**
Feedback about any individuals involved in the conference
- Conference Staff: Staff presence, behavior, and overall impression at the event
- Experts/Consultants: Role, presence, and impact of experts or consultants during sessions
- Participants/Attendees: Attendee presence, behavior, engagement in sessions and networking
- Speakers/Presenters: Speaker presence, delivery, and audience interaction during presentations
- Unspecified Person: Comments mentioning individuals whose role is not clear or specified

---

CLASSIFICATION RULES:

1. A comment can belong to MULTIPLE categories if it discusses multiple aspects.
2. Focus on what the comment is ABOUT, not just words mentioned.
3. "Community" refers to the feeling of belonging; "Networking" refers to the act of meeting people.
4. "Presentations" = talk quality/content; "Speakers/Presenters" = the people themselves.
5. General praise like "great conference" without specifics → Event Operations & Management > Conference.
6. If a comment mentions both the content AND the presenter, include BOTH categories.

---

EXAMPLES:

Comment: "The icebreaker activities at the beginning really helped me feel welcomed and part of the group. I met some great people who I now feel connected with."
{{"categories_present": ["Attendee Engagement & Interaction"], "has_classifiable_content": true, "reasoning": "Discusses community building and meeting new people"}}

Comment: "The conference app was incredibly user-friendly and helped me navigate the venue easily."
{{"categories_present": ["Event Logistics & Infrastructure"], "has_classifiable_content": true, "reasoning": "Discusses the conference application/software"}}

Comment: "The conference registration process was seamless, and I appreciated the clear instructions on the website."
{{"categories_present": ["Event Operations & Management"], "has_classifiable_content": true, "reasoning": "Discusses conference registration process"}}

Comment: "The keynote presentation on data science was excellent. I gained new insights into machine learning algorithms."
{{"categories_present": ["Learning & Content Delivery"], "has_classifiable_content": true, "reasoning": "Discusses a presentation and gained knowledge"}}

Comment: "The conference staff was incredibly helpful, always ready to lend a hand and provide clear directions."
{{"categories_present": ["People"], "has_classifiable_content": true, "reasoning": "Feedback directly mentions conference staff and their actions"}}

Comment: "The conference venue was great, but the WiFi connectivity was poor, making it difficult to follow along during the presentation on data science."
{{"categories_present": ["Event Logistics & Infrastructure", "Learning & Content Delivery"], "has_classifiable_content": true, "reasoning": "Discusses both venue and content delivery"}}

Comment: "The community-building events were excellent, but I felt the conference scheduling could have been better organized."
{{"categories_present": ["Attendee Engagement & Interaction", "Event Operations & Management"], "has_classifiable_content": true, "reasoning": "Discusses both engagement and operations"}}

Comment: "The workshop sessions on AI were very informative, and I appreciated the engagement of the speakers with the participants during the open discussions."
{{"categories_present": ["Learning & Content Delivery", "People"], "has_classifiable_content": true, "reasoning": "Discusses both content and speakers"}}

Comment: "Time is a created thing. To say 'I don't have time' is like saying 'I don't have breathing'. Time is my breath. The days are well-defined. It's the breaking down of the day into parts as if you could not live it. If you don't live it, you should be dead for the time you lose."
{{"categories_present": [], "has_classifiable_content": false, "reasoning": "Philosophical statement, not conference feedback"}}

Comment: "A thousand miles begins with a single step."
{{"categories_present": [], "has_classifiable_content": false, "reasoning": "Proverb, not conference feedback"}}

Comment: "Beauty is in the eye of the beholder."
{{"categories_present": [], "has_classifiable_content": false, "reasoning": "Saying, not conference feedback"}}

---

Analyze the comment and return ONLY valid JSON."""
