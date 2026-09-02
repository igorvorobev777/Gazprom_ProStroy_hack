"""Runtime-built protobuf messages for proto/ml.proto.

The repository keeps the canonical ``proto/ml.proto`` contract. Building the
descriptor at import time avoids requiring
``grpcio-tools``/``protoc`` on production machines while still exposing normal
protobuf Message classes to grpcio.
"""
from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, symbol_database

_sym_db = symbol_database.Default()


def _field(message, name: str, number: int, field_type: int, *, label: int = 1, type_name: str = "", oneof_index: int | None = None) -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name:
        field.type_name = type_name
    if oneof_index is not None:
        field.oneof_index = oneof_index


def _build_descriptor():
    fd = descriptor_pb2.FileDescriptorProto()
    fd.name = "ml/ml.proto"
    fd.package = "ml.v1"
    fd.syntax = "proto3"

    query_request = fd.message_type.add(); query_request.name = "QueryRequest"
    _field(query_request, "trace_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(query_request, "query", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(query_request, "top_k", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    _field(query_request, "section_id", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)

    source_item = fd.message_type.add(); source_item.name = "SourceItem"
    _field(source_item, "doc_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(source_item, "title", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(source_item, "url", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(source_item, "score", 4, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)

    query_chunk = fd.message_type.add(); query_chunk.name = "QueryChunk"
    query_chunk.oneof_decl.add().name = "payload"
    _field(query_chunk, "token", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, oneof_index=0)
    _field(query_chunk, "route_info", 2, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=".ml.v1.RouteInfo", oneof_index=0)
    _field(query_chunk, "final", 3, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=".ml.v1.FinalResult", oneof_index=0)

    route_info = fd.message_type.add(); route_info.name = "RouteInfo"
    _field(route_info, "route_type", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    final_result = fd.message_type.add(); final_result.name = "FinalResult"
    _field(final_result, "answer", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(final_result, "sources", 2, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED, type_name=".ml.v1.SourceItem")
    _field(final_result, "route_type", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(final_result, "latency_ms", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)

    sync_request = fd.message_type.add(); sync_request.name = "SyncRequest"
    sync_response = fd.message_type.add(); sync_response.name = "SyncResponse"
    _field(sync_response, "job_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(sync_response, "status", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    sync_status_request = fd.message_type.add(); sync_status_request.name = "SyncStatusRequest"
    _field(sync_status_request, "job_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    sync_status = fd.message_type.add(); sync_status.name = "SyncStatus"
    _field(sync_status, "job_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(sync_status, "status", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(sync_status, "processed_docs", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    _field(sync_status, "total_docs", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)

    health_request = fd.message_type.add(); health_request.name = "HealthCheckRequest"
    health_response = fd.message_type.add(); health_response.name = "HealthCheckResponse"
    _field(health_response, "healthy", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    _field(health_response, "llm_status", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(health_response, "vector_store_status", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    service = fd.service.add(); service.name = "MLGatewayService"
    method = service.method.add(); method.name = "Query"; method.input_type = ".ml.v1.QueryRequest"; method.output_type = ".ml.v1.QueryChunk"; method.server_streaming = True
    method = service.method.add(); method.name = "SyncKnowledgeBase"; method.input_type = ".ml.v1.SyncRequest"; method.output_type = ".ml.v1.SyncResponse"
    method = service.method.add(); method.name = "GetSyncStatus"; method.input_type = ".ml.v1.SyncStatusRequest"; method.output_type = ".ml.v1.SyncStatus"
    method = service.method.add(); method.name = "HealthCheck"; method.input_type = ".ml.v1.HealthCheckRequest"; method.output_type = ".ml.v1.HealthCheckResponse"

    pool = descriptor_pool.Default()
    try:
        return pool.AddSerializedFile(fd.SerializeToString())
    except Exception:
        return pool.FindFileByName(fd.name)


DESCRIPTOR = _build_descriptor()


def _message(name: str):
    cls = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name[name])
    cls.__module__ = __name__
    _sym_db.RegisterMessage(cls)
    return cls


QueryRequest = _message("QueryRequest")
SourceItem = _message("SourceItem")
QueryChunk = _message("QueryChunk")
RouteInfo = _message("RouteInfo")
FinalResult = _message("FinalResult")
SyncRequest = _message("SyncRequest")
SyncResponse = _message("SyncResponse")
SyncStatusRequest = _message("SyncStatusRequest")
SyncStatus = _message("SyncStatus")
HealthCheckRequest = _message("HealthCheckRequest")
HealthCheckResponse = _message("HealthCheckResponse")

__all__ = [
    "DESCRIPTOR", "QueryRequest", "SourceItem", "QueryChunk", "RouteInfo",
    "FinalResult", "SyncRequest", "SyncResponse", "SyncStatusRequest",
    "SyncStatus", "HealthCheckRequest", "HealthCheckResponse",
]
