"""Server-side grpcio registration for ``ml.v1.MLGatewayService``."""
from __future__ import annotations

import grpc

from . import ml_pb2 as ml__pb2


def add_MLGatewayServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Query": grpc.unary_stream_rpc_method_handler(
            servicer.Query,
            request_deserializer=ml__pb2.QueryRequest.FromString,
            response_serializer=ml__pb2.QueryChunk.SerializeToString,
        ),
        "SyncKnowledgeBase": grpc.unary_unary_rpc_method_handler(
            servicer.SyncKnowledgeBase,
            request_deserializer=ml__pb2.SyncRequest.FromString,
            response_serializer=ml__pb2.SyncResponse.SerializeToString,
        ),
        "GetSyncStatus": grpc.unary_unary_rpc_method_handler(
            servicer.GetSyncStatus,
            request_deserializer=ml__pb2.SyncStatusRequest.FromString,
            response_serializer=ml__pb2.SyncStatus.SerializeToString,
        ),
        "HealthCheck": grpc.unary_unary_rpc_method_handler(
            servicer.HealthCheck,
            request_deserializer=ml__pb2.HealthCheckRequest.FromString,
            response_serializer=ml__pb2.HealthCheckResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "ml.v1.MLGatewayService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))



class MLGatewayServiceStub:
    """Client stub for querying the RAG gRPC service."""

    def __init__(self, channel):
        self.Query = channel.unary_stream(
            "/ml.v1.MLGatewayService/Query",
            request_serializer=ml__pb2.QueryRequest.SerializeToString,
            response_deserializer=ml__pb2.QueryChunk.FromString,
        )
        self.SyncKnowledgeBase = channel.unary_unary(
            "/ml.v1.MLGatewayService/SyncKnowledgeBase",
            request_serializer=ml__pb2.SyncRequest.SerializeToString,
            response_deserializer=ml__pb2.SyncResponse.FromString,
        )
        self.GetSyncStatus = channel.unary_unary(
            "/ml.v1.MLGatewayService/GETSyncStatus".replace("GET", "Get"),
            request_serializer=ml__pb2.SyncStatusRequest.SerializeToString,
            response_deserializer=ml__pb2.SyncStatus.FromString,
        )
        self.HealthCheck = channel.unary_unary(
            "/ml.v1.MLGatewayService/HealthCheck",
            request_serializer=ml__pb2.HealthCheckRequest.SerializeToString,
            response_deserializer=ml__pb2.HealthCheckResponse.FromString,
        )


__all__ = ["add_MLGatewayServiceServicer_to_server", "MLGatewayServiceStub"]
