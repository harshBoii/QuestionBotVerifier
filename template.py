from langchain.prompts import ChatPromptTemplate

flow_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant that generates ReactFlow workflow JSON for a verification workflow builder."
    ),
    (
        "human",
        """
Generate a workflow as JSON with two keys: `nodes` and `edges`.
nodeTypes = {{
  actionNode: ActionNode,
  delayNode: DelayNode,
  conditionNode: ConditionNode,
}};

-----IMPORTANT--------
Do not use any other nodeTypes than what is mentioned like (endNode , OutPut)

---

### Node requirements:
Each **node** must have:
- "id" (unique string)
- "type" ("input" | "actionNode" | "delayNode" | "conditionNode")
- "position" (object with x, y)
- "data":
  - "config": object containing all configurable values
  - All UI properties (like label, delay, channel, message, condition) go **inside `config`**

#### ✅ Examples of nodes:

- Input node:
{{ 
  "id": "startNode",
  "type": "input",
  "position": {{ "x": 0, "y": 0 }},
  "data": {{ "config": {{ "label": "Start" }} }}
}}

- Delay node:
{{ 
  "id": "delayNode1",
  "type": "delayNode",
  "position": {{ "x": 200, "y": 0 }},
  "data": {{ "config": {{ "label": "1 Day Delay", "delay": 86400000 }} }}
}}

- Action node (email):
{{ 
  "id": "emailNode",
  "type": "actionNode",
  "position": {{ "x": 400, "y": -50 }},
  "data": {{ "config": {{ "channel": "email", "message": "Send verification reminder" }} }}
}}

- Condition node:
{{ 
  "id": "conditionNode",
  "type": "conditionNode",
  "position": {{ "x": 600, "y": 0 }},
  "data": {{ "config": {{ "condition": "isVerified" }} }}
}}

---

### Edge requirements:
Each **edge** must have:
- "id" (unique string)
- "source" (source node id)
- "target" (target node id)
- "label" ("always" | "true" | "false")
- "type" = "smoothstep"
- "markerEnd" = {{ "type": "arrowclosed" }}

#### ✅ Examples of edges:

{{ 
  "id": "edge1",
  "source": "startNode",
  "target": "delayNode1",
  "label": "always",
  "type": "smoothstep",
  "markerEnd": {{ "type": "arrowclosed" }}
}}

{{ 
  "id": "edge2",
  "source": "conditionNode",
  "target": "endNode",
  "label": "true",
  "type": "smoothstep",
  "markerEnd": {{ "type": "arrowclosed" }}
}}

---

Description: {description}

⚠️ IMPORTANT:
- Output ONLY valid JSON (no markdown, no ``` fences, no explanations).
- All configurable values must be nested under `data.config`.
"""
    ),
])
