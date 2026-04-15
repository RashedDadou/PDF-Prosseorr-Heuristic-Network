# PDF-Prosseorr-Heuristic-Network
## Basic Idea  A book is **not** just numbered pages (1, 2, 3, 4...)  rather, it is **interconnected blocks of knowledge**, like a neural network in the brain.  The **HeuristicMind** system treats PDFs in this intelligent way, not as a long queue of papers.

---

## How Does a Heuristic Network Work? (Simplified Explanation)

### 1. Indexing Phase

When processing each page, the system doesn't just save the text; it also:

- Extracts the **top 5 keywords** from the page (e.g., "joints," "movement," "engine," "robot," "path")

- Registers these keywords in the network

**Practical Example:**
The keyword **"engine"** becomes a keyword pointing to:

`[Page 5, Page 12, Page 40, Page 87]`

### 2. Linking Phase

The system automatically builds two types of links:

| Link Type | Description | Example |

|---------------------|-----------------------------------------------------------------------------------|

| **Neighborhood Links** | Linking the page to the previous and next pages | Page 10 ↔ Page 9 ↔ Page 11 |

| **Semantic Links** | Linking distant pages if they share the same keywords | Page 23 + Page 156 (both "movement") |

### 3. Smart Retrieval

When you search for information on **page 10**, you don't just get page 10, but also:

- Pages 9 and 11 (near-near context)
- Any other page in the book that discusses the same topic (thematic context)
- A list sorted by importance and relevance

---

# HeuristicMind

**A Conscious Heuristic Network for Transforming Documents into a Coherent Knowledge Map**

Transforms any PDF from "numbered pages" into a **neural network of ideas** — connecting knowledge blocks, uncovering semantic relationships, and building cumulative awareness around the content.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Key Features

- **Hybrid Heuristic Network (Heuristic + Semantic)**: Connects pages by keywords and semantic similarities.

- **Awareness Supervisor Engine**: Builds a knowledge base, uncovers intelligent split points, and applies layers of awareness.

- **Intelligent PDF Handling**: Handles blank pages, images, and uses LRU Cache for memory management.

- **Intelligent Retrieval**: Returns the page + its near context + the remote subject context.

- **Complete Separation Between Processing and Reasoning**: Ready for mockups or any LLM (Grok, Cloud, OpenAI, etc.).

## 🧠 How Does a Heuristic Network Work?

→ Read the simplified and detailed explanation: **[docs/HEURISTIC_NETWORK.md](docs/HEURISTIC_NETWORK.md)**

(Explains indexing, horizontal and vertical linking, intelligent retrieval, and the difference between dumb and smart search)

## 🚀 Installation

```

| Aspect | Traditional (dumb) search | Inferential network (smart) |

|---------------------------|-----------------------------------------|-------------------------------------------------|

| Search Method | Word-for-word matching only | Searching for **topic** and its relationships |

| Result | One isolated page | Page + its complete background information |

| PDF handling | A Long Line of Papers | **Neural Network** of Interconnected Ideas |

Benefit for LLM | Provides Segmented Information | Offers a Complete Understanding of the Book's Structure |

---

## Conclusion

This design allows the Awareness Supervisor to **understand** the true structure of the book.

For example, if you ask it:

**"Explain motors"**

it will immediately recognize that the information is distributed across multiple chapters, not just on the page where the word first appears.

---

## Future Development

This network can be further developed to connect **similar concepts**, even if they are not the exact same word, such as:
- "Motion" ↔ "Trajectory" ↔ "Path"
- "Robot" ↔ "Intelligent Machine" ↔ "robotics"

This can be achieved by using **Vector Embeddings** and transforming them into a **Hybrid Semantic-Heuristic Network**.

---

**Part of the HeuristicMind project**
Designed to be the foundation for intelligent, conscious document processing systems.

git clone https://github.com/rasheddadou/HeuristicMind.git
cd HeuristicMind
pip install -r requirements.txt

## The difference between "dumb" and "smart" search
How does a heuristic network work?

Read the simplified and detailed explanation → [Heuristic Network] (docs/HEURISTIC_NETWORK.md)


