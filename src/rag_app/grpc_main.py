from __future__ import annotations

import asyncio
import logging

import grpc

from rag_orchestrator import get_settings
from rag_transport.proto import ml_pb2_grpc

from .composition import resolve_paths
from .grpc_service import GRPC_OPTIONS, MLGatewayService, RagRuntime


async def serve() -> None:
    settings = resolve_paths(get_settings())
    runtime = RagRuntime(settings)
    await runtime.initialize()

    server = grpc.aio.server(options=GRPC_OPTIONS)
    ml_pb2_grpc.add_MLGatewayServiceServicer_to_server(
        MLGatewayService(runtime), server
    )
    address = f"{settings.grpc_host}:{settings.grpc_port}"
    bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"Could not bind gRPC server to {address}")

    await server.start()
    manifest = runtime.current_container().index.manifest
    logging.getLogger(__name__).info(
        "RAG gRPC ready on %s (documents=%d, chunks=%d)",
        address,
        manifest.document_count,
        manifest.chunk_count,
    )
    print(
        f"RAG gRPC ready: {address} | documents={manifest.document_count} "
        f"chunks={manifest.chunk_count}",
        flush=True,
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
