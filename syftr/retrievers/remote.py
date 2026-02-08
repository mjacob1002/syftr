"""Remote retriever that calls external retrieval APIs."""
import typing as T
import requests
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from syftr.logger import logger


class RemoteRetriever(BaseRetriever):
    """Retriever that calls an external retrieval API.

    Args:
        api_url: Base URL of the retrieval API (e.g., "http://130.127.134.41:6002")
        retrieval_method: Type of retrieval ("dense_small", "dense_large", "bm25", "hybrid_small", "hybrid_large")
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        api_url: str,
        retrieval_method: str = "dense_small",
        top_k: int = 10,
        timeout: int = 30,
        **kwargs: T.Any,
    ) -> None:
        """Initialize remote retriever."""
        super().__init__(**kwargs)
        self.api_url = api_url.rstrip("/")
        self.retrieval_method = retrieval_method
        self.top_k = top_k
        self.timeout = timeout

        logger.info(
            f"Initialized RemoteRetriever: {retrieval_method} at {api_url}, top_k={top_k}"
        )

    def _retrieve(self, query_bundle: QueryBundle) -> T.List[NodeWithScore]:
        """Retrieve nodes for a query by calling external API.

        Args:
            query_bundle: Query to search for

        Returns:
            List of nodes with scores
        """
        query_str = query_bundle.query_str

        # Prepare request
        url = f"{self.api_url}/search"
        payload = {
            "query": query_str,
            "k": self.top_k
        }

        try:
            logger.debug(f"Calling retrieval API: {url} with query: {query_str[:100]}...")
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            logger.info(
                f"Retrieved {len(results)} documents from {self.retrieval_method} "
                f"(requested top_k={self.top_k})"
            )

            # Convert API response to NodeWithScore objects
            nodes_with_scores = []
            for idx, result in enumerate(results):
                score = result.get("score", 0.0)
                document = result.get("document", {})

                # Extract content and metadata
                page_content = document.get("page_content", "")
                metadata = document.get("metadata", {})

                # Create TextNode
                node = TextNode(
                    text=page_content,
                    metadata=metadata,
                    id_=f"{self.retrieval_method}_{idx}_{hash(page_content)}"
                )

                # Create NodeWithScore
                node_with_score = NodeWithScore(
                    node=node,
                    score=score
                )
                nodes_with_scores.append(node_with_score)

            return nodes_with_scores

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling retrieval API: {e}")
            # Return empty list on error
            return []
        except Exception as e:
            logger.error(f"Unexpected error in remote retrieval: {e}")
            return []


# Mapping of retrieval methods to ports
RETRIEVAL_SERVER_PORTS = {
    "bm25": 6030,
    "dense_small": 6002,
    "dense_large": 6001,
    "hybrid_small": 6024,
    "hybrid_large": 6025,
}


def get_remote_retriever(
    host: str,
    retrieval_method: str,
    top_k: int = 10,
    timeout: int = 30,
) -> RemoteRetriever:
    """Factory function to create a remote retriever.

    Args:
        host: IP address of retrieval server
        retrieval_method: Type of retrieval (bm25, dense_small, dense_large, hybrid_small, hybrid_large)
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds

    Returns:
        Configured RemoteRetriever instance
    """
    if retrieval_method not in RETRIEVAL_SERVER_PORTS:
        raise ValueError(
            f"Unknown retrieval method: {retrieval_method}. "
            f"Valid options: {list(RETRIEVAL_SERVER_PORTS.keys())}"
        )

    port = RETRIEVAL_SERVER_PORTS[retrieval_method]
    api_url = f"http://{host}:{port}"

    return RemoteRetriever(
        api_url=api_url,
        retrieval_method=retrieval_method,
        top_k=top_k,
        timeout=timeout,
    )
