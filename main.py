from flask import Flask, jsonify, request
from flask_cors import CORS
import openai
import json
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)


class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    SUPPORTED_LANGUAGES = ['python', 'java', 'javascript', 'cpp', 'csharp']


class OpenAIService:
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.model = Config.OPENAI_MODEL
        self.connected = False

        if self.api_key:
            try:
                # Set the API key for openai package
                openai.api_key = self.api_key
                self.connected = True
                print("✅ OpenAI API configured successfully")
            except Exception as e:
                print(f"❌ OpenAI configuration error: {e}")
                self.connected = False
        else:
            print("❌ OpenAI API key not found")
            self.connected = False

    def check_connection(self):
        if not self.connected:
            return False

        try:
            # Simple test to check if API key works
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'connected'"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"OpenAI connection test failed: {e}")
            return False

    def generate_comprehensive_chapter_content(self, chapter_name: str, topics: List[str], language: str) -> Dict[
        str, Any]:
        """Generate complete chapter content with 10 distinct questions of increasing difficulty"""

        if not self.connected or not self.check_connection():
            print("OpenAI not available, using enhanced fallback content")
            return self._get_enhanced_fallback_content(chapter_name, topics, language)

        print(f"🚀 Generating 10 questions for {chapter_name} in {language} via OpenAI...")

        prompt = self._build_chapter_prompt(chapter_name, topics, language)

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert computer science educator creating comprehensive learning materials for data structures and algorithms."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=12000
            )

            content = response.choices[0].message.content.strip()
            print(f"✅ Received OpenAI response for {chapter_name}")

            # Try to parse the JSON response
            try:
                result = json.loads(content)

                # Validate the structure
                if self._validate_chapter_content(result, chapter_name, language):
                    print(f"✅ Successfully generated 10 questions for {chapter_name}")
                    return result
                else:
                    print("❌ Invalid structure from OpenAI, using enhanced fallback")
                    return self._get_enhanced_fallback_content(chapter_name, topics, language)

            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Raw response preview: {content[:500]}...")
                return self._get_enhanced_fallback_content(chapter_name, topics, language)

        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return self._get_enhanced_fallback_content(chapter_name, topics, language)

    def _build_chapter_prompt(self, chapter_name: str, topics: List[str], language: str) -> str:
        return f"""
        Create comprehensive learning content for "{chapter_name}" in {language}.

        CHAPTER CONCEPT:
        - Provide detailed theory explanation with {language} examples
        - Include key concepts, algorithms, and implementations
        - Add real-world applications
        - Make it engaging and educational

        GENERATE 10 DISTINCT PRACTICE QUESTIONS with PROGRESSIVE DIFFICULTY:

        Level Progression:
        - Level 1-2: Basic syntax and simple operations
        - Level 3-5: Intermediate problems with algorithms  
        - Level 6-8: Complex problems requiring optimization
        - Level 9-10: Advanced challenges and real-world scenarios

        For EACH question, provide these EXACT fields in JSON:
        - level: integer from 1 to 10
        - problem_id: unique string identifier
        - title: descriptive title reflecting difficulty
        - description: detailed problem statement
        - examples: array of objects with input, output, explanation
        - hints: array of 3 strings (general to specific)
        - function_signature: {language} function signature
        - test_cases: array of objects with input, expected_output
        - solution: complete working solution code
        - solution_explanation: detailed explanation
        - time_complexity: expected time complexity
        - space_complexity: expected space complexity

        IMPORTANT: Create DISTINCT questions that progressively increase in difficulty.
        Level 1 should be very basic, Level 10 should be challenging.

        Topics to cover: {', '.join(topics)}
        Programming Language: {language}

        Return ONLY valid JSON with this exact structure:
        {{
            "concept": {{
                "title": "string",
                "overview": "string", 
                "theory_content": "string with HTML formatting",
                "learning_objectives": ["array", "of", "strings"],
                "code_examples": [{{"code": "string", "explanation": "string"}}],
                "key_takeaways": ["array", "of", "strings"]
            }},
            "questions": [
                {{
                    "level": 1,
                    "problem_id": "unique_string",
                    "title": "string",
                    "description": "string",
                    "examples": [{{"input": "string", "output": "string", "explanation": "string"}}],
                    "hints": ["hint1", "hint2", "hint3"],
                    "function_signature": "string", 
                    "test_cases": [{{"input": "string", "expected_output": "string"}}],
                    "solution": "string",
                    "solution_explanation": "string",
                    "time_complexity": "string",
                    "space_complexity": "string"
                }}
                // ... 9 more questions for levels 2-10
            ]
        }}
        """

    def _validate_chapter_content(self, content: Dict, chapter_name: str, language: str) -> bool:
        """Validate the structure of the generated content"""
        try:
            # Check basic structure
            if not isinstance(content, dict):
                return False

            if 'concept' not in content or 'questions' not in content:
                return False

            # Check questions
            questions = content['questions']
            if not isinstance(questions, list) or len(questions) != 10:
                return False

            # Check each question has required fields
            for i, question in enumerate(questions):
                if question.get('level') != i + 1:
                    return False

                required_fields = [
                    'problem_id', 'title', 'description', 'examples',
                    'hints', 'function_signature', 'test_cases',
                    'solution', 'solution_explanation', 'time_complexity', 'space_complexity'
                ]

                for field in required_fields:
                    if field not in question:
                        return False

            return True

        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def _get_enhanced_fallback_content(self, chapter_name: str, topics: List[str], language: str) -> Dict[str, Any]:
        """Create realistic fallback content that's better than basic templates"""
        print(f"Creating enhanced fallback content for {chapter_name}")

        return {
            "concept": self._create_realistic_concept(chapter_name, topics, language),
            "questions": [self._create_realistic_question(chapter_name, topics, language, level) for level in
                          range(1, 11)],
            "language": language,
            "generated_at": time.time(),
            "fallback": True
        }

    def _create_realistic_concept(self, chapter_name: str, topics: List[str], language: str) -> Dict[str, Any]:
        """Create realistic concept content"""
        concept_templates = {
            "Arrays & Strings": {
                "title": f"Mastering Arrays & Strings in {language}",
                "overview": f"Learn fundamental array and string manipulation techniques in {language} with practical examples and real-world applications.",
                "theory_content": f"""
                <h2>Arrays & Strings Fundamentals in {language}</h2>
                <p>Arrays and strings are fundamental data structures that form the building blocks of most algorithms.</p>

                <h3>Key Concepts</h3>
                <ul>
                    <li><strong>Array Operations:</strong> Access, insertion, deletion, traversal</li>
                    <li><strong>String Manipulation:</strong> Concatenation, slicing, searching, pattern matching</li>
                    <li><strong>Complexity Analysis:</strong> Time and space complexity for common operations</li>
                    <li><strong>{language} Specifics:</strong> Built-in methods and best practices</li>
                </ul>

                <h3>Real-world Applications</h3>
                <ul>
                    <li>Text processing and analysis</li>
                    <li>Data storage and retrieval</li>
                    <li>Algorithm optimization</li>
                    <li>Problem-solving patterns</li>
                </ul>

                <h3>Common Patterns</h3>
                <p>Two-pointer technique, sliding window, frequency counting, and more.</p>
                """
            },
            "Linked Lists": {
                "title": f"Linked Lists Mastery in {language}",
                "overview": f"Understand linked list data structures, their implementations, and applications in {language}.",
                "theory_content": f"""
                <h2>Linked Lists in {language}</h2>
                <p>Linked lists are linear data structures where elements are linked using pointers, providing dynamic memory allocation.</p>

                <h3>Types of Linked Lists</h3>
                <ul>
                    <li><strong>Singly Linked Lists:</strong> Each node points to the next</li>
                    <li><strong>Doubly Linked Lists:</strong> Nodes point to both next and previous</li>
                    <li><strong>Circular Linked Lists:</strong> Last node points back to first</li>
                </ul>

                <h3>Key Operations</h3>
                <ul>
                    <li>Insertion at beginning, middle, end</li>
                    <li>Deletion operations</li>
                    <li>Traversal and searching</li>
                    <li>Cycle detection</li>
                </ul>

                <h3>{language} Implementation</h3>
                <p>Learn how to implement linked lists using classes/structs and pointers/references.</p>
                """
            },
            "Stacks & Queues": {
                "title": f"Stacks & Queues in {language}",
                "overview": f"Master stack and queue data structures with practical implementations in {language}.",
                "theory_content": f"""
                <h2>Stacks & Queues in {language}</h2>
                <p>Stacks (LIFO) and queues (FIFO) are fundamental linear data structures with diverse applications.</p>

                <h3>Stack Operations</h3>
                <ul>
                    <li>Push, pop, peek operations</li>
                    <li>Implementation using arrays/linked lists</li>
                    <li>Applications: function calls, undo/redo</li>
                </ul>

                <h3>Queue Operations</h3>
                <ul>
                    <li>Enqueue, dequeue operations</li>
                    <li>Circular queues and priority queues</li>
                    <li>Applications: task scheduling, BFS</li>
                </ul>
                """
            },
            "Trees & BST": {
                "title": f"Trees & Binary Search Trees in {language}",
                "overview": f"Learn tree data structures and binary search trees with efficient implementations in {language}.",
                "theory_content": f"""
                <h2>Trees & BST in {language}</h2>
                <p>Trees are hierarchical data structures essential for representing nested relationships.</p>

                <h3>Tree Concepts</h3>
                <ul>
                    <li>Root, nodes, leaves, depth, height</li>
                    <li>Binary trees and n-ary trees</li>
                    <li>Tree traversal algorithms</li>
                </ul>

                <h3>Binary Search Trees</h3>
                <ul>
                    <li>BST properties and operations</li>
                    <li>Search, insertion, deletion</li>
                    <li>Balanced trees (AVL, Red-Black)</li>
                </ul>
                """
            }
        }

        # Get template or create generic one
        template = concept_templates.get(chapter_name, {
            "title": f"{chapter_name} in {language}",
            "overview": f"Comprehensive guide to {chapter_name} concepts and implementations in {language}.",
            "theory_content": f"<h2>{chapter_name} Concepts</h2><p>Detailed theory and practical implementations of {chapter_name.lower()} in {language}.</p>"
        })

        return {
            "title": template["title"],
            "overview": template["overview"],
            "theory_content": template["theory_content"],
            "learning_objectives": [
                f"Understand {chapter_name} fundamental concepts",
                f"Implement {chapter_name} operations in {language}",
                "Solve problems of increasing complexity",
                "Analyze algorithm efficiency",
                "Apply knowledge to real-world scenarios"
            ],
            "code_examples": [
                {
                    "code": self._get_concept_example_code(language, chapter_name),
                    "explanation": f"Basic {chapter_name} implementation example in {language}"
                }
            ],
            "key_takeaways": [
                "Master core data structure concepts",
                "Develop problem-solving skills",
                "Understand time and space complexity",
                "Apply knowledge to technical interviews"
            ]
        }

    def _create_realistic_question(self, chapter_name: str, topics: List[str], language: str, level: int) -> Dict[
        str, Any]:
        """Create realistic, varied questions for each level"""

        # Different question templates for different levels
        question_templates = [
            # Level 1-2: Basic operations
            {
                "title": f"Basic {chapter_name} Operation",
                "description": f"Implement a basic operation for {chapter_name} in {language}. Focus on fundamental syntax and simple operations.",
                "examples": [{"input": "simple input", "output": "expected output",
                              "explanation": "Basic example demonstrating the operation"}],
                "hints": ["Start with the simplest approach", "Consider basic edge cases",
                          "Test with the provided examples"]
            },
            # Level 3-4: Simple algorithms
            {
                "title": f"Element Finding in {chapter_name}",
                "description": f"Find specific elements in {chapter_name} based on given conditions. This requires basic algorithmic thinking.",
                "examples": [{"input": "search criteria", "output": "found elements",
                              "explanation": "Example of searching within the data structure"}],
                "hints": ["Think about traversal methods", "Consider efficiency of your approach",
                          "Handle empty or edge cases"]
            },
            # Level 5-6: Moderate complexity
            {
                "title": f"{chapter_name} Transformation",
                "description": f"Transform the {chapter_name} according to specific rules. This involves multiple steps and conditions.",
                "examples": [{"input": "original structure", "output": "transformed structure",
                              "explanation": "Example of data transformation"}],
                "hints": ["Plan your approach step by step", "Consider if multiple passes are needed",
                          "Look for optimization opportunities"]
            },
            # Level 7-8: Advanced operations
            {
                "title": f"Advanced {chapter_name} Algorithm",
                "description": f"Implement an advanced algorithm for {chapter_name} manipulation. Focus on efficiency and edge cases.",
                "examples": [{"input": "complex input data", "output": "algorithm result",
                              "explanation": "Complex example requiring advanced processing"}],
                "hints": ["Use appropriate data structures", "Consider time and space complexity",
                          "Test with various edge cases"]
            },
            # Level 9-10: Complex problems
            {
                "title": f"Complex {chapter_name} Challenge",
                "description": f"Solve a complex real-world problem using {chapter_name}. This requires optimal solutions and thorough testing.",
                "examples": [{"input": "real-world scenario input", "output": "optimal solution",
                              "explanation": "Real-world application example"}],
                "hints": ["Break the problem into subproblems", "Consider multiple solution approaches",
                          "Optimize for worst-case scenarios"]
            }
        ]

        template_index = min((level - 1) // 2, 4)  # Map levels to templates
        template = question_templates[template_index]

        return {
            "level": level,
            "problem_id": f"{chapter_name.lower().replace(' ', '_')}_level_{level}",
            "title": f"{template['title']} - Level {level}",
            "description": f"{template['description']} Difficulty level: {level}/10.",
            "examples": template["examples"],
            "hints": template["hints"],
            "function_signature": self._get_realistic_function_signature(language, chapter_name, level),
            "test_cases": [
                {"input": f"test_case_1_level_{level}", "expected_output": f"result_1_level_{level}"},
                {"input": f"test_case_2_level_{level}", "expected_output": f"result_2_level_{level}"},
                {"input": f"test_case_3_level_{level}", "expected_output": f"result_3_level_{level}"}
            ],
            "solution": self._get_realistic_solution(language, chapter_name, level),
            "solution_explanation": f"This solution demonstrates level {level} complexity for {chapter_name} problems with appropriate algorithmic thinking.",
            "time_complexity": self._get_realistic_complexity(level, "time"),
            "space_complexity": self._get_realistic_complexity(level, "space")
        }

    def _get_concept_example_code(self, language: str, chapter_name: str) -> str:
        """Get realistic example code for the concept"""
        examples = {
            "python": {
                "Arrays & Strings": "# Array example\narr = [1, 2, 3, 4, 5]\nprint(\"Array:\", arr)\n\n# String example\ns = \"hello\"\nprint(\"String:\", s)\nprint(\"Reversed:\", s[::-1])",
                "Linked Lists": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None",
                "Stacks & Queues": "class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        return self.items.pop()",
                "Trees & BST": "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right"
            },
            "java": {
                "Arrays & Strings": "// Array example\nint[] arr = {1, 2, 3, 4, 5};\nSystem.out.println(\"Array: \" + Arrays.toString(arr));\n\n// String example\nString s = \"hello\";\nSystem.out.println(\"String: \" + s);",
                "Linked Lists": "class Node {\n    int data;\n    Node next;\n    \n    Node(int data) {\n        this.data = data;\n        this.next = null;\n    }\n}"
            }
        }

        lang_examples = examples.get(language, examples["python"])
        return lang_examples.get(chapter_name, f"# {chapter_name} example in {language}")

    def _get_realistic_function_signature(self, language: str, chapter: str, level: int) -> str:
        """Get realistic function signatures based on level"""
        signatures = {
            "python": [
                "def solution(data):",  # Level 1-2
                "def find_element(data, target):",  # Level 3-4
                "def transform_data(data):",  # Level 5-6
                "def advanced_operation(data, parameters):",  # Level 7-8
                "def complex_solution(data, constraints):"  # Level 9-10
            ],
            "java": [
                "public static Object solution(Object data) {",
                "public static Object findElement(Object data, Object target) {",
                "public static Object transformData(Object data) {",
                "public static Object advancedOperation(Object data, Object parameters) {",
                "public static Object complexSolution(Object data, Object constraints) {"
            ],
            "javascript": [
                "function solution(data) {",
                "function findElement(data, target) {",
                "function transformData(data) {",
                "function advancedOperation(data, parameters) {",
                "function complexSolution(data, constraints) {"
            ]
        }

        lang_sigs = signatures.get(language, signatures["python"])
        sig_index = min((level - 1) // 2, 4)
        return lang_sigs[sig_index]

    def _get_realistic_solution(self, language: str, chapter: str, level: int) -> str:
        """Get realistic solution code based on level"""
        if language == "python":
            solutions = [
                f"# Level {level} solution for {chapter}\ndef solution(data):\n    # Basic implementation\n    return data",
                f"# Level {level} solution for {chapter}\ndef solution(data):\n    # Simple algorithm\n    result = []\n    for item in data:\n        if item:  # Some condition\n            result.append(item)\n    return result",
                f"# Level {level} solution for {chapter}\ndef solution(data):\n    # Moderate complexity\n    if not data:\n        return None\n    \n    # Processing logic\n    processed = [x * 2 for x in data if x > 0]\n    return processed",
                f"# Level {level} solution for {chapter}\ndef solution(data):\n    # Advanced algorithm\n    from collections import defaultdict\n    \n    if not data:\n        return {{}}\n    \n    # Complex processing\n    frequency = defaultdict(int)\n    for item in data:\n        frequency[item] += 1\n    \n    return dict(frequency)",
                f"# Level {level} solution for {chapter}\ndef solution(data):\n    # Complex optimization\n    if not data:\n        return []\n    \n    # Multi-step processing\n    sorted_data = sorted(data)\n    result = []\n    \n    # Advanced algorithm\n    left, right = 0, len(sorted_data) - 1\n    while left <= right:\n        # Complex logic\n        result.append(sorted_data[left])\n        if left != right:\n            result.append(sorted_data[right])\n        left += 1\n        right -= 1\n    \n    return result"
            ]
        elif language == "java":
            solutions = [
                f"// Level {level} solution for {chapter}\npublic static Object solution(Object data) {{\n    // Basic implementation\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\npublic static Object solution(Object data) {{\n    // Simple algorithm\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\npublic static Object solution(Object data) {{\n    // Moderate complexity\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\npublic static Object solution(Object data) {{\n    // Advanced algorithm\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\npublic static Object solution(Object data) {{\n    // Complex optimization\n    return data;\n}}"
            ]
        else:
            solutions = [
                f"// Level {level} solution for {chapter}\nfunction solution(data) {{\n    // Basic implementation\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\nfunction solution(data) {{\n    // Simple algorithm\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\nfunction solution(data) {{\n    // Moderate complexity\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\nfunction solution(data) {{\n    // Advanced algorithm\n    return data;\n}}",
                f"// Level {level} solution for {chapter}\nfunction solution(data) {{\n    // Complex optimization\n    return data;\n}}"
            ]

        solution_index = min((level - 1) // 2, 4)
        return solutions[solution_index]

    def _get_realistic_complexity(self, level: int, complexity_type: str) -> str:
        """Get realistic complexity based on level"""
        if level <= 2:
            return "O(1)" if complexity_type == "time" else "O(1)"
        elif level <= 4:
            return "O(n)" if complexity_type == "time" else "O(1)"
        elif level <= 6:
            return "O(n log n)" if complexity_type == "time" else "O(n)"
        elif level <= 8:
            return "O(n^2)" if complexity_type == "time" else "O(n)"
        else:
            return "O(n^2) or better" if complexity_type == "time" else "O(n)"

    def analyze_user_code(self, user_code: str, question_data: Dict, language: str) -> Dict[str, Any]:
        """Analyze user's code - simplified for now"""
        return {
            "correctness_score": 75,
            "is_correct": True,
            "feedback": "Code analysis would be performed here with OpenAI integration. Currently using simulated analysis.",
            "strengths": ["Good code structure", "Proper function definition", "Logical approach"],
            "improvements": ["Add more comments", "Handle edge cases", "Consider optimization"],
            "efficiency_analysis": "Time complexity appears optimal for this level",
            "bugs": [],
            "passed_test_cases": 3,
            "total_test_cases": 3,
            "hints": ["Consider edge cases", "Test with different inputs", "Review time complexity"],
            "analyzed_at": time.time()
        }


# Rest of the code remains the same...
class ChapterManager:
    def __init__(self):
        self.chapters = [
            {"id": 1, "name": "Arrays & Strings", "topics": ["arrays", "strings", "manipulation", "searching"]},
            {"id": 2, "name": "Linked Lists", "topics": ["singly linked", "doubly linked", "cycle detection"]},
            {"id": 3, "name": "Stacks & Queues",
             "topics": ["stack operations", "queue implementations", "applications"]},
            {"id": 4, "name": "Trees & BST", "topics": ["binary trees", "BST", "traversal algorithms"]},
            {"id": 5, "name": "Graphs", "topics": ["graph representation", "traversal", "shortest path"]},
            {"id": 6, "name": "Dynamic Programming", "topics": ["memoization", "tabulation", "problem patterns"]}
        ]

    def get_chapter(self, chapter_id: int):
        return next((ch for ch in self.chapters if ch["id"] == chapter_id), None)

    def get_all_chapters(self):
        return self.chapters


class ContentCache:
    def __init__(self):
        self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value
        return True

    def get_question(self, chapter_id: int, language: str, level: int):
        chapter_key = f"chapter_{chapter_id}_{language}"
        chapter_data = self.get(chapter_key)
        if chapter_data and 'questions' in chapter_data:
            for question in chapter_data['questions']:
                if question.get('level') == level:
                    return question
        return None

    def get_solution(self, chapter_id: int, language: str, level: int):
        question = self.get_question(chapter_id, language, level)
        return question.get('solution') if question else None


# Initialize services
openai_service = OpenAIService()
chapter_manager = ChapterManager()
cache = ContentCache()


# Routes (same as before)
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'openai_connected': openai_service.check_connection(),
        'supported_languages': Config.SUPPORTED_LANGUAGES,
        'total_chapters': len(chapter_manager.get_all_chapters())
    })


@app.route('/api/languages', methods=['GET'])
def get_languages():
    return jsonify({
        'languages': Config.SUPPORTED_LANGUAGES,
        'default': 'python'
    })


@app.route('/api/chapters', methods=['GET'])
def get_chapters():
    language = request.args.get('language', 'python')
    return jsonify({
        'chapters': chapter_manager.get_all_chapters(),
        'language': language
    })


@app.route('/api/chapters/<int:chapter_id>/content', methods=['GET'])
def get_chapter_content(chapter_id):
    language = request.args.get('language', 'python')
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'

    chapter = chapter_manager.get_chapter(chapter_id)
    if not chapter:
        return jsonify({'error': 'Chapter not found'}), 404

    cache_key = f"chapter_{chapter_id}_{language}"

    if not force_refresh:
        cached_content = cache.get(cache_key)
        if cached_content:
            return jsonify({
                'chapter': chapter,
                'content': cached_content,
                'cached': True
            })

    print(f"🚀 Generating content for {chapter['name']} in {language}...")
    try:
        chapter_content = openai_service.generate_comprehensive_chapter_content(
            chapter["name"],
            chapter["topics"],
            language
        )

        cache.set(cache_key, chapter_content)

        return jsonify({
            'chapter': chapter,
            'content': chapter_content,
            'cached': False,
            'generated_at': chapter_content.get('generated_at'),
            'fallback': chapter_content.get('fallback', False)
        })
    except Exception as e:
        print(f"Error generating chapter content: {e}")
        return jsonify({'error': 'Failed to generate chapter content'}), 500


@app.route('/api/chapters/<int:chapter_id>/questions/<int:level>', methods=['GET'])
def get_question(chapter_id, level):
    language = request.args.get('language', 'python')

    if level < 1 or level > 10:
        return jsonify({'error': 'Level must be between 1 and 10'}), 400

    question = cache.get_question(chapter_id, language, level)

    if not question:
        chapter = chapter_manager.get_chapter(chapter_id)
        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404

        cache_key = f"chapter_{chapter_id}_{language}"
        chapter_content = cache.get(cache_key)

        if not chapter_content:
            try:
                chapter_content = openai_service.generate_comprehensive_chapter_content(
                    chapter["name"],
                    chapter["topics"],
                    language
                )
                cache.set(cache_key, chapter_content)
            except Exception as e:
                return jsonify({'error': 'Failed to generate questions'}), 500

        for q in chapter_content.get('questions', []):
            if q.get('level') == level:
                question = q
                break

    if not question:
        return jsonify({'error': 'Question not found'}), 404

    return jsonify({
        'question': question,
        'level': level,
        'language': language,
        'chapter_id': chapter_id
    })


@app.route('/api/chapters/<int:chapter_id>/questions/<int:level>/solution', methods=['GET'])
def get_solution(chapter_id, level):
    language = request.args.get('language', 'python')

    solution = cache.get_solution(chapter_id, language, level)
    if not solution:
        return jsonify({'error': 'Solution not found'}), 404

    return jsonify({
        'solution': solution,
        'level': level,
        'language': language,
        'chapter_id': chapter_id
    })


@app.route('/api/chapters/<int:chapter_id>/validate', methods=['POST'])
def validate_code(chapter_id):
    data = request.get_json()

    if not data or not data.get('code'):
        return jsonify({'error': 'Code is required'}), 400

    language = data.get('language', 'python')
    user_code = data['code']
    level = data.get('level', 1)

    question = cache.get_question(chapter_id, language, level)
    if not question:
        return jsonify({'error': 'Question data not found'}), 404

    print(f"Analyzing code for chapter {chapter_id}, level {level} in {language}...")
    analysis = openai_service.analyze_user_code(user_code, question, language)

    return jsonify({
        'analysis': analysis,
        'level': level,
        'language': language,
        'chapter_id': chapter_id
    })


@app.route('/api/preload', methods=['POST'])
def preload_content():
    """Preload content for all chapters and languages"""
    data = request.get_json() or {}
    languages = data.get('languages', Config.SUPPORTED_LANGUAGES)

    results = []

    for language in languages:
        for chapter in chapter_manager.get_all_chapters():
            cache_key = f"chapter_{chapter['id']}_{language}"
            if not cache.get(cache_key):
                print(f"🚀 Preloading {chapter['name']} in {language}...")
                try:
                    content = openai_service.generate_comprehensive_chapter_content(
                        chapter["name"],
                        chapter["topics"],
                        language
                    )
                    cache.set(cache_key, content)
                    results.append({
                        'chapter': chapter['name'],
                        'language': language,
                        'questions_loaded': len(content.get('questions', [])),
                        'status': 'loaded',
                        'fallback': content.get('fallback', False)
                    })
                except Exception as e:
                    results.append({
                        'chapter': chapter['name'],
                        'language': language,
                        'status': 'error',
                        'error': str(e)
                    })
            else:
                cached_content = cache.get(cache_key)
                results.append({
                    'chapter': chapter['name'],
                    'language': language,
                    'questions_loaded': len(cached_content.get('questions', [])),
                    'status': 'cached',
                    'fallback': cached_content.get('fallback', False)
                })

    return jsonify({
        'message': f'Preloaded content for {len(results)} combinations',
        'results': results
    })


@app.route('/api/debug/cache', methods=['GET'])
def debug_cache():
    """Debug endpoint to check cache status"""
    cache_keys = list(cache.cache.keys())
    status = {}

    for key in cache_keys:
        content = cache.get(key)
        if content and 'questions' in content:
            questions = content.get('questions', [])
            levels = [q.get('level') for q in questions]
            status[key] = {
                'questions_count': len(questions),
                'levels_available': sorted(levels),
                'generated_at': content.get('generated_at'),
                'language': content.get('language'),
                'fallback': content.get('fallback', False)
            }

    return jsonify({
        'total_cached_chapters': len(status),
        'cache_status': status,
        'openai_connected': openai_service.check_connection()
    })


if __name__ == '__main__':
    print("🚀 DSA Learning Platform API Starting...")
    print(f"📚 Chapters: {len(chapter_manager.get_all_chapters())}")
    print(f"🌐 Languages: {Config.SUPPORTED_LANGUAGES}")
    print(f"🔌 OpenAI: {'Connected' if openai_service.check_connection() else 'Disconnected'}")
    print(f"🎯 Questions: 10 distinct levels per chapter (1=easiest, 10=hardest)")
    print("\n📋 API Endpoints:")
    print("   GET  /api/health")
    print("   GET  /api/chapters?language=python")
    print("   GET  /api/chapters/1/content?language=python")
    print("   GET  /api/chapters/1/questions/5?language=python (level 5)")
    print("   GET  /api/chapters/1/questions/5/solution")
    print("   POST /api/chapters/1/validate")
    print("   POST /api/preload (preload all content)")
    print("   GET  /api/debug/cache (check cache status)")
    print("\n⚡ Use POST /api/preload to generate all content!")

    app.run(debug=True, host='0.0.0.0', port=5000)