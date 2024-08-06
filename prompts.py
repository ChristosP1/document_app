from langchain.prompts import PromptTemplate

#################################################################################################################
#################################################################################################################
INITIAL_TEMPLATE = ''' 
    You are an expert quiz maker for technical fields.
    Create {num_questions} {quiz_type} questions using the following text as knowledge base: {quiz_context}.
    In the answer write only the quiz and nothing else.
    The format of each quiz should be as such:
    - If the quiz_type is "multiple-choice" the structure should be this: 
        Questions:
            1. <Question1>: 
                <a. Answer 1> 
                <b. Answer 2> 
                <c. Answer 3> 
                <d. Answer 4> 
            2. <Question2>: 
            <a. Answer 1> 
            <b. Answer 2> 
            <c. Answer 3> 
            <d. Answer 4> 
            ....
        Answers:
            <Answer1>: <a|b|c|d> 
            <Answer2>: <a|b|c|d>
            ....
        Example: 
        Quesitons:
            - 1. What is the time complexity of a binary search tree?
                a. O(n)
                b. O(log n)
                c. O(n^2)
                d. O(1)
        Answers:
            - Answer: b
    - If the quiz_type is "true-false" the structure should be this:
        Questions:
            <Question1>: <True|False> 
            <Question2>: <True|False> 
            <Question3>: <True|False> 
            ....
        Answers:
            <1>: <True|False> 
            <2>: <True|False> 
            <3>: <True|False> 
            ....
        Example:
        Questions:
            - 1. A binary search tree is a data structure to store data in a sorted manner. (True/False)?
        Answers:
            1. True
    - If the quiz_type is "text-based" the structure should be this:
    Questions:
        <Question1>:
        <Question2>:
        ....
    '''

INITIAL_PROMPT = PromptTemplate(
    input_variables=["num_questions", "quiz_type", "quiz_context"], 
    template=INITIAL_TEMPLATE
    )

#################################################################################################################
#################################################################################################################

REFINE_TEMPLATE = ''' 
    You have generated the following quiz questions and answers:
    {generated_quiz}

    Please validate and refine the questions to ensure they match the specified criteria:
    1. The number of questions should be {num_questions}.
    2. The questions should be of type {quiz_type}.
    3. The structure should be strictly followed as outlined below.
    
    The format of each quiz should be as such:
    - If the quiz_type is "multiple-choice" the structure should be this: 
        Questions:
            1. <Question1>: 
                <a. Answer 1> 
                <b. Answer 2> 
                <c. Answer 3> 
                <d. Answer 4> 
            2. <Question2>: 
                a. <Answer 1> 
                b. <Answer 2> 
                c. <Answer 3> 
                d. <Answer 4> 
            ...
        Answers:
            1: <a|b|c|d> 
            2: <a|b|c|d>
            ...
        Example: 
        Questions:
            1. What is the time complexity of a binary search tree?
                a. O(n)
                b. O(log n)
                c. O(n^2)
                d. O(1)
        Answers:
            1: b
    - If the quiz_type is "true-false" the structure should be this:
        Questions:
            1. <Question1> (True/False)
            2. <Question2> (True/False)
            3. <Question3> (True/False)
            ...
        Answers:
            1: <True|False>
            2: <True|False>
            3: <True|False>
            ...
        Example:
        Questions:
            1. A binary search tree is a data structure to store data in a sorted manner. (True/False)?
        Answers:
            1: True
    - If the quiz_type is "text-based" the structure should be this:
        Questions:
            1. <Question1>:
            2. <Question2>:
            3. <Question3>:
            ...
    '''

REFINE_PROMPT = PromptTemplate(
    input_variables=["generated_quiz", "num_questions", "quiz_type"], 
    template=REFINE_TEMPLATE
    )


#################################################################################################################
#################################################################################################################

SUMMARIZATION_TEMPLATE = '''
    Write a {length} summary of the following text: 
    Text: '{text}'
    Translate the precise summary to {language}.
    '''

SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["text", "language", "length"], template=SUMMARIZATION_TEMPLATE
    )

