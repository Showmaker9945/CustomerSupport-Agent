"""
Knowledge base for customer support FAQs.

Provides vector-based FAQ search using ChromaDB with category filtering
and confidence scoring.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class FAQResult:
    """Result from FAQ search."""
    question: str
    answer: str
    category: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation for display."""
        return (
            f"Q: {self.question}\n"
            f"A: {self.answer}\n"
            f"Category: {self.category} | Confidence: {self.confidence:.2%}"
        )


class FAQStore:
    """
    Knowledge base for customer support FAQs.

    Provides vector-based semantic search using ChromaDB and
    sentence transformers for embeddings.
    """

    # Sample FAQs for a generic SaaS product
    SAMPLE_FAQS = [
        {
            "question": "How do I reset my password?",
            "answer": "Go to Settings > Security > Password and click 'Reset Password'. "
                     "You'll receive an email with a secure link to create a new password. "
                     "The link expires in 24 hours.",
            "category": "account",
            "keywords": ["password", "reset", "login", "forgot"]
        },
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept all major credit cards (Visa, MasterCard, American Express), "
                     "PayPal, and wire transfers for annual plans. For enterprise customers, "
                     "we also offer invoicing with NET-30 terms.",
            "category": "billing",
            "keywords": ["payment", "credit card", "paypal", "invoice"]
        },
        {
            "question": "How do I cancel my subscription?",
            "answer": "You can cancel anytime from your account settings. Go to Settings > "
                     "Billing > Subscription and click 'Cancel Plan'. Your access continues "
                     "until the end of your current billing period. No refunds for partial months.",
            "category": "billing",
            "keywords": ["cancel", "subscription", "refund", "delete"]
        },
        {
            "question": "Can I upgrade or downgrade my plan?",
            "answer": "Yes! You can change your plan at any time. Upgrades take effect immediately, "
                     "and you'll be charged the prorated difference. Downgrades take effect at "
                     "the next billing cycle.",
            "category": "billing",
            "keywords": ["upgrade", "downgrade", "plan", "change"]
        },
        {
            "question": "How do I add team members?",
            "answer": "Go to Settings > Team > Members and click 'Invite Member'. Enter their "
                     "email address and select their role (Admin, Editor, or Viewer). They'll "
                     "receive an invitation link to join your workspace.",
            "category": "workspace",
            "keywords": ["team", "member", "invite", "collaborator"]
        },
        {
            "question": "What's the difference between Viewer, Editor, and Admin?",
            "answer": "Viewers can only view content. Editors can create and edit content but "
                     "can't change settings. Admins have full access including billing, team "
                     "management, and all settings.",
            "category": "workspace",
            "keywords": ["role", "permission", "admin", "editor", "viewer"]
        },
        {
            "question": "Is my data secure?",
            "answer": "Yes. We use AES-256 encryption for data at rest and TLS 1.3 for data "
                     "in transit. We're SOC 2 Type II certified and GDPR compliant. We also "
                     "offer optional two-factor authentication (2FA).",
            "category": "security",
            "keywords": ["security", "encryption", "gdpr", "soc2", "privacy"]
        },
        {
            "question": "Do you offer an API?",
            "answer": "Yes! We offer a REST API with comprehensive documentation. You can generate "
                     "API keys in Settings > Developer. API access is available on Pro and "
                     "Enterprise plans.",
            "category": "technical",
            "keywords": ["api", "integration", "developer", "webhook"]
        },
        {
            "question": "How do I export my data?",
            "answer": "Go to Settings > Data > Export and choose your format (CSV, JSON, or PDF). "
                     "For large datasets, we'll email you when the export is ready. You can also "
                     "use our API for automated exports.",
            "category": "technical",
            "keywords": ["export", "download", "backup", "data"]
        },
        {
            "question": "What are your system requirements?",
            "answer": "Our web app works on any modern browser (Chrome, Firefox, Safari, Edge). "
                     "For mobile, we support iOS 14+ and Android 10+. We also offer desktop "
                     "apps for macOS 11+ and Windows 10+.",
            "category": "technical",
            "keywords": ["requirements", "browser", "system", "compatible"]
        },
        {
            "question": "How do I contact support?",
            "answer": "You can reach us via live chat (bottom right of your screen), email at "
                     "support@example.com, or by submitting a ticket through the Help Center. "
                     "Pro and Enterprise customers get priority 24/7 support.",
            "category": "support",
            "keywords": ["contact", "help", "phone", "email", "chat"]
        },
        {
            "question": "What is your response time for support tickets?",
            "answer": "Free plan: 48 hours. Pro plan: 24 hours. Enterprise: 4 hours. Critical "
                     "issues for Enterprise customers are handled within 1 hour. Live chat is "
                     "available 24/7 for paid plans.",
            "category": "support",
            "keywords": ["response", "time", "sla", "wait"]
        },
        {
            "question": "Can I try before I buy?",
            "answer": "Absolutely! We offer a 14-day free trial of our Pro plan with full "
                     "features. No credit card required. At the end of the trial, you can "
                     "choose a plan or continue with our free tier.",
            "category": "billing",
            "keywords": ["trial", "free", "demo", "test"]
        },
        {
            "question": "Do you offer discounts for nonprofits or education?",
            "answer": "Yes! We offer 50% off for registered nonprofits and educational institutions. "
                     "Contact our sales team with proof of status to receive your discount code.",
            "category": "billing",
            "keywords": ["discount", "nonprofit", "education", "student"]
        },
        {
            "question": "How often do you release updates?",
            "answer": "We release small updates weekly and major features quarterly. All updates "
                     "are automatic with no downtime. You can view our roadmap and upcoming "
                     "features at example.com/roadmap.",
            "category": "product",
            "keywords": ["update", "release", "new", "feature", "roadmap"]
        },
        {
            "question": "Can I integrate with other tools?",
            "answer": "We integrate with 100+ tools including Slack, Zapier, Google Workspace, "
                     "Microsoft 365, Salesforce, and more. See our full list at example.com/integrations. "
                     "Custom integrations available for Enterprise.",
            "category": "technical",
            "keywords": ["integration", "slack", "zapier", "connect", "sync"]
        },
        {
            "question": "What happens if I exceed my storage limit?",
            "answer": "You'll receive a notification at 80% and 95% of your limit. If you exceed it, "
                     "new uploads will be paused but existing data remains accessible. You can "
                     "upgrade your plan or purchase additional storage ($5/GB/month).",
            "category": "billing",
            "keywords": ["storage", "limit", "quota", "upgrade"]
        },
        {
            "question": "How do I enable two-factor authentication?",
            "answer": "Go to Settings > Security > Two-Factor Authentication. Choose between an "
                     "authenticator app (Google Authenticator, Authy) or SMS. Scan the QR code "
                     "or enter your phone number to complete setup.",
            "category": "security",
            "keywords": ["2fa", "two-factor", "authentication", "security"]
        },
        {
            "question": "Can I transfer my account to another owner?",
            "answer": "Yes. Account transfers are available for Enterprise plans. Contact support "
                     "to initiate the transfer. The new owner will receive an invitation, and "
                     "both parties must confirm the transfer.",
            "category": "account",
            "keywords": ["transfer", "ownership", "sell", "account"]
        },
        {
            "question": "Where are your servers located?",
            "answer": "We have data centers in the US (East/West), EU (Frankfurt), and Asia "
                     "(Singapore). You can choose your preferred region during signup or change "
                     "it later in Settings > Data > Region.",
            "category": "technical",
            "keywords": ["server", "location", "region", "data center"]
        }
    ]

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "faq_knowledge_base"
    ):
        """
        Initialize the FAQ store.

        Args:
            chroma_path: Path to persist ChromaDB data
            embedding_model: Name of sentence-transformers model
            collection_name: Name of ChromaDB collection
        """
        self.chroma_path = chroma_path or settings.chroma_persist_dir
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.collection_name = collection_name

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Customer support FAQ knowledge base"}
            )

            logger.info(f"ChromaDB initialized at {self.chroma_path}")

            # Load sample FAQs if collection is empty
            if self.collection.count() == 0:
                logger.info("Initializing with sample FAQs...")
                self._load_sample_faqs()

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text."""
        return self.embedding_model.encode(text).tolist()

    def _load_sample_faqs(self) -> None:
        """Load sample FAQs into the knowledge base."""
        for faq in self.SAMPLE_FAQS:
            self.add_faq(
                question=faq["question"],
                answer=faq["answer"],
                category=faq["category"],
                metadata={"keywords": faq.get("keywords", [])}
            )
        logger.info(f"Loaded {len(self.SAMPLE_FAQS)} sample FAQs")

    def add_faq(
        self,
        question: str,
        answer: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a FAQ entry to the knowledge base.

        Args:
            question: FAQ question
            answer: FAQ answer
            category: FAQ category
            metadata: Optional metadata (keywords, priority, etc.)

        Returns:
            FAQ ID
        """
        try:
            # Combine question and answer for better search
            combined_text = f"Question: {question}\nAnswer: {answer}"

            # Create embedding
            embedding = self._create_embedding(combined_text)

            # Prepare metadata (ChromaDB doesn't allow lists)
            faq_metadata = {
                "category": category,
                "question": question,
            }

            # Add metadata but convert lists to strings
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, list):
                        faq_metadata[key] = ",".join(str(v) for v in value)
                    else:
                        faq_metadata[key] = value

            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[faq_metadata],
                ids=[f"faq_{hash(question + answer)}"]
            )

            logger.debug(f"Added FAQ: {question[:50]}...")
            return f"faq_{hash(question + answer)}"

        except Exception as e:
            logger.error(f"Failed to add FAQ: {e}")
            raise

    def load_faqs_from_file(self, file_path: str) -> int:
        """
        Bulk load FAQs from a file.

        Args:
            file_path: Path to JSON or CSV file

        Returns:
            Number of FAQs loaded

        Expected JSON format:
        [
            {
                "question": "...",
                "answer": "...",
                "category": "...",
                "metadata": {...}
            },
            ...
        ]
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"FAQ file not found: {file_path}")

        try:
            if path.suffix == ".json":
                with open(path, 'r', encoding='utf-8') as f:
                    faqs = json.load(f)
            elif path.suffix == ".csv":
                import csv
                faqs = []
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        faqs.append({
                            "question": row.get("question", ""),
                            "answer": row.get("answer", ""),
                            "category": row.get("category", "general"),
                            "metadata": {"keywords": row.get("keywords", "").split(",")}
                                if row.get("keywords") else {}
                        })
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Add FAQs
            count = 0
            errors = []
            for faq in faqs:
                try:
                    self.add_faq(
                        question=faq["question"],
                        answer=faq["answer"],
                        category=faq.get("category", "general"),
                        metadata=faq.get("metadata")
                    )
                    count += 1
                except Exception as e:
                    # Log individual failures but continue loading remaining FAQs
                    logger.warning(f"Failed to load FAQ: {faq.get('question', 'unknown')}: {e}")
                    errors.append({"faq": faq.get('question', 'unknown'), "error": str(e)})

            if errors:
                logger.warning(f"Loaded {count}/{len(faqs)} FAQs successfully. {len(errors)} failed.")

            logger.info(f"Loaded {count} FAQs from {file_path}")
            return count

        except Exception as e:
            logger.error(f"Failed to load FAQs from file: {e}")
            raise

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 3,
        min_confidence: float = 0.0
    ) -> List[FAQResult]:
        """
        Search for relevant FAQs.

        Args:
            query: Search query
            category: Optional category filter
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            List of FAQResults sorted by confidence
        """
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)

            # Build where clause for category filter
            where_clause = {"category": category} if category else None

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Get more results to filter
                where=where_clause
            )

            # Process results
            faq_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    # Convert distance to confidence (ChromaDB uses L2 distance)
                    # L2 distance ranges from 0 (identical) to infinity
                    # We'll map to confidence using a simple formula
                    confidence = max(0, 1 - (distance / 2))

                    # Apply confidence threshold
                    if confidence < min_confidence:
                        continue

                    # Extract question from metadata or doc
                    question = metadata.get("question", "")
                    if not question and doc:
                        # Parse question from document
                        lines = doc.split('\n')
                        for line in lines:
                            if line.startswith("Question:"):
                                question = line.replace("Question:", "").strip()
                                break

                    faq_results.append(FAQResult(
                        question=question or "Unknown question",
                        answer=metadata.get("answer", doc),
                        category=metadata.get("category", "general"),
                        confidence=confidence,
                        metadata={
                            k: v for k, v in metadata.items()
                            if k not in ["question", "answer", "category"]
                        }
                    ))

            # Sort by confidence and limit to top_k
            faq_results.sort(key=lambda x: x.confidence, reverse=True)
            return faq_results[:top_k]

        except Exception as e:
            logger.error(f"FAQ search failed: {e}")
            return []

    def get_categories(self) -> List[str]:
        """
        Get all unique FAQ categories.

        Returns:
            List of category names
        """
        try:
            # Get all documents to extract categories
            results = self.collection.get(include=["metadatas"])

            categories = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "category" in metadata:
                        categories.add(metadata["category"])

            return sorted(list(categories))

        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAQ store.

        Returns:
            Dictionary with stats
        """
        try:
            total_count = self.collection.count()
            categories = self.get_categories()

            return {
                "total_faqs": total_count,
                "categories": categories,
                "category_count": len(categories),
                "embedding_model": self.embedding_model_name,
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def delete_faq(self, question: str) -> bool:
        """
        Delete a FAQ by question.

        Args:
            question: Question text to match

        Returns:
            True if deleted, False if not found
        """
        try:
            # Search for FAQs with matching question in metadata
            # Get all and filter client-side since ChromaDB's where clause
            # has limited exact match capabilities
            all_results = self.collection.get(include=["metadatas"])

            matching_ids = []
            for i, metadata in enumerate(all_results.get("metadatas", [])):
                if metadata.get("question") == question:
                    matching_ids.append(all_results["ids"][i])

            if matching_ids:
                self.collection.delete(ids=matching_ids)
                logger.info(f"Deleted {len(matching_ids)} FAQ(s): {question}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete FAQ: {e}")
            return False

    def clear_all(self) -> None:
        """Clear all FAQs from the store."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Customer support FAQ knowledge base"}
            )
            logger.info("Cleared all FAQs")
        except Exception as e:
            logger.error(f"Failed to clear FAQs: {e}")
            raise


def create_faq_store(
    chroma_path: Optional[Path] = None,
    load_samples: bool = True
) -> FAQStore:
    """
    Factory function to create and initialize an FAQStore.

    Args:
        chroma_path: Optional custom path for ChromaDB
        load_samples: Whether to load sample FAQs

    Returns:
        Initialized FAQStore instance
    """
    store = FAQStore(chroma_path=chroma_path)

    if not load_samples:
        store.clear_all()

    return store


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    """Demonstrate FAQ store usage."""
    print("=" * 60)
    print("FAQ Store Demo")
    print("=" * 60)

    # Create FAQ store
    store = create_faq_store()

    # Get stats
    stats = store.get_stats()
    print(f"\nFAQ Store Stats:")
    print(f"  Total FAQs: {stats['total_faqs']}")
    print(f"  Categories: {', '.join(stats['categories'])}")
    print(f"  Embedding Model: {stats['embedding_model']}")

    # Search examples
    queries = [
        "reset password",
        "payment methods",
        "cancel subscription"
    ]

    print("\nSearch Results:")
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = store.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result.question}")
            print(f"       Confidence: {result.confidence:.1%}")

    print("\n" + "=" * 60)
