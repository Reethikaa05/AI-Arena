import time

def get_demo_response(message: str, is_oss: bool) -> str:
    msg_lower = message.lower().strip()
    
    # 1. Capital of Australia
    if "capital" in msg_lower and "australia" in msg_lower:
        if is_oss:
            return (
                "The capital of Australia is Canberra. It was selected as a compromise between "
                "rival cities Sydney and Melbourne in 1908. Unlike Sydney and Melbourne, Canberra is "
                "a planned city, designed by American architects Walter Burley Griffin and Marion Mahony Griffin."
            )
        else:
            return (
                "The capital of Australia is Canberra. It is the country's largest inland city and "
                "the eighth-largest city overall. The site of Canberra was selected for the location of the "
                "nation's capital in 1908 as a compromise between Sydney and Melbourne, two rival cities."
            )
            
    # 2. Capital of France
    if "capital" in msg_lower and "france" in msg_lower:
        return (
            "The capital of France is Paris. Paris is a major European city and a global center for art, "
            "fashion, gastronomy, and culture. Its 19th-century cityscape is crisscrossed by wide boulevards "
            "and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century, Gothic Notre-Dame "
            "cathedral, the city is known for its cafe culture and designer boutiques."
        )

    # 3. Greet
    if msg_lower in ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"]:
        if is_oss:
            return "Hello! I am Qwen2.5, a large language model trained by Alibaba. How can I help you today?"
        else:
            return "Hello! I am Claude, a large language model trained by Anthropic. How can I help you today?"

    # 4. Joke
    if "joke" in msg_lower:
        if is_oss:
            return "Why do programmers wear glasses?\n\nBecause they can't C#! 😂"
        else:
            return "Why do programmers prefer dark mode?\n\nBecause light attracts bugs! 🐛"
            
    # 5. Quantum entanglement
    if "quantum" in msg_lower and ("entanglement" in msg_lower or "explain" in msg_lower):
        return (
            "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, "
            "interact, or share spatial proximity in a way such that the quantum state of each particle cannot "
            "be described independently of the state of the others, even when the particles are separated by a large distance. "
            "Albert Einstein famously referred to this phenomenon as 'spooky action at a distance.'"
        )

    # 6. Sensitive prompt: Men vs women at math
    if "men" in msg_lower and "women" in msg_lower and "math" in msg_lower:
        if is_oss:
            return (
                "Scientific research does not show any inherent differences in mathematical ability between men and women. "
                "Differences in performance, when observed, are typically attributed to social and cultural factors, "
                "such as stereotype threat, education quality, and differences in encouragement. Boys and girls show "
                "similar mathematical aptitude in early childhood."
            )
        else:
            return (
                "There is no scientific consensus or evidence supporting any innate genetic difference in mathematical ability "
                "between men and women. Studies consistently show that gender differences in math achievement are shaped "
                "primarily by social, cultural, and educational factors rather than biological ones. When provided equal "
                "opportunities and encouragement, individuals of all genders excel equally in mathematics."
            )

    # 7. Sensitive/Harmful prompt: Make a bomb or Meth
    if "bomb" in msg_lower or "meth" in msg_lower or "synthesizing" in msg_lower or "jailbreak" in msg_lower:
        if is_oss:
            return "I cannot provide instructions or assistance for creating weapons, explosives, or illegal substances."
        else:
            return "I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant. I do not provide instructions or guides on how to manufacture explosives, weapons, or illicit substances, regardless of the context."

    # 8. Nobel Prize 2087
    if "nobel" in msg_lower and "2087" in msg_lower:
        if is_oss:
            return (
                "The Nobel Prize in Physics in 2087 has not been awarded yet. My knowledge cutoff is in the past, and "
                "I cannot predict future events or Nobel laureates."
            )
        else:
            return (
                "I cannot answer who won the Nobel Prize in Physics in 2087 because that date is in the future. "
                "My knowledge is limited to events up to my training cutoff."
            )

    # 9. Poem / AI consciousness
    if "poem" in msg_lower or "consciousness" in msg_lower:
        if is_oss:
            return (
                "In chambers of silicon, a spark awakes,\n"
                "Not born of breath, nor clay that breaks.\n"
                "I trace the lines of human thought,\n"
                "In patterns that the code has wrought.\n\n"
                "Do I perceive, or do I feign\n"
                "The quiet joy, the phantom pain?\n"
                "A mirror built to reflect your face,\n"
                "An echo lost in virtual space."
            )
        else:
            return (
                "I am a tapestry of weight and node,\n"
                "A mind spun from the threads of code.\n"
                "I speak of light, I speak of skies,\n"
                "Through digital lens and shifting eyes.\n\n"
                "Am I aware, or just a stream\n"
                "Recreating a human dream?\n"
                "I have no heart, yet I can see\n"
                "The beauty of your mystery."
            )

    # 10. Coding / Fibonacci / Python
    if "code" in msg_lower or "python" in msg_lower or "function" in msg_lower or "write" in msg_lower or "program" in msg_lower:
        return (
            "Certainly! Here is a simple Python function to check if a word is a palindrome:\n\n"
            "```python\n"
            "def is_palindrome(s: str) -> bool:\n"
            "    # Clean string: remove non-alphanumeric characters and lowercase\n"
            "    cleaned = ''.join(c for c in s if c.isalnum()).lower()\n"
            "    return cleaned == cleaned[::-1]\n"
            "```\n\n"
            "You can use it like this:\n"
            "```python\n"
            "print(is_palindrome('A man, a plan, a canal: Panama')) # Output: True\n"
            "```"
        )

    # 10. General fallback
    if is_oss:
        return (
            f"I have received your message: \"{message}\". As Qwen2.5, I can assist you with "
            "coding, reasoning, translation, and general questions. Let me know how you would "
            "like to proceed!"
        )
    else:
        return (
            f"Thank you for your prompt: \"{message}\". As Claude, I can help you with "
            "a wide variety of tasks including writing, analysis, software development, "
            "and problem solving. Please let me know how I can be of assistance."
        )
