
import Scene.SceneTypes;
import Scene.Scene;

[[vk::binding(0, 1)]]
ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);

StructuredBuffer<GeometryInstanceData> instanceDescBuffer;
StructuredBuffer<MeshDesc> meshDescBuffer;

StaticVertexData load_vertex(MeshDesc mesh_data, uint vertex_id)
{
    var vb_offset = mesh_data.vbOffset;
    var normal_offset = mesh_data.normalOffset;
    var bindlessIndex = mesh_data.bindlessIndex;

    ByteAddressBuffer vertex_buffer = t_BindlessBuffers[bindlessIndex];

    if (mesh_data.ibOffset > 0) {
        vertex_id = vertex_buffer.Load(mesh_data.ibOffset + vertex_id * 4);
    }

    var id = vertex_id;
    float3 pos = asfloat(vertex_buffer.Load3(3 * vertex_id * 4 + vb_offset));
    float3 normal =
        asfloat(vertex_buffer.Load3(3 * vertex_id * 4 + normal_offset));
    float2 texCrd = asfloat(
        vertex_buffer.Load2(2 * vertex_id * 4 + mesh_data.texCrdOffset));

    StaticVertexData data =
        StaticVertexData(pos, normal, float4(0, 0, 0, 0), texCrd, 0);
    return data;
}

VertexInfo get_interpolated_vertex(
    uint instance_id,
    uint primitive_id,
    float2 baricentric2)
{
    GeometryInstanceData instance = instanceDescBuffer[instance_id];
    MeshDesc mesh = meshDescBuffer[instance.geometryID];
    var id_in_indexbuffer = mesh.ibOffset + primitive_id * 3 * 4;

    ByteAddressBuffer vertex_buffer = t_BindlessBuffers[mesh.bindlessIndex];
    uint i0 = primitive_id * 3;
    uint i1 = primitive_id * 3 + 1;
    uint i2 = primitive_id * 3 + 2;

    StaticVertexData v[3];

    v[0] = load_vertex(mesh, i0);
    v[1] = load_vertex(mesh, i1);
    v[2] = load_vertex(mesh, i2);

    float4x4 worldMat = transpose(instance.transform);
    float3x3 worldInvTransposeMat = transpose(inverse(float3x3(worldMat)));

    float3 baricentric = float3(
        1.0 - baricentric2.x - baricentric2.y, baricentric2.x, baricentric2.y);
    return computeVertexData(
        v, worldMat, worldInvTransposeMat, false, baricentric);
}
