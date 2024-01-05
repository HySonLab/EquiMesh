# # ----------------------------------------------------------------------------
# # -                        Open3D: www.open3d.org                            -
# # ----------------------------------------------------------------------------
# # Copyright (c) 2018-2023 www.open3d.org
# # SPDX-License-Identifier: MIT
# # ----------------------------------------------------------------------------

# import open3d as o3d
# import numpy as np
# import open3d
# # from colordict import ColorDict


# device = o3d.core.Device("CPU:0")
# dtype_f = o3d.core.float32
# dtype_i = o3d.core.int32

# # COLOR = np.array(list(ColorDict().values()))[:,:3]
# # print(COLOR.shape)


# if __name__ == "__main__":
#     o3d.visualization.webrtc_server.enable_webrtc()


#     bunny = o3d.data.BunnyMesh()
#     omesh = o3d.io.read_triangle_mesh(bunny.path)
#     omesh = omesh.simplify_quadric_decimation(target_number_of_triangles=1700)
#     mesh = o3d.t.geometry.TriangleMesh.from_legacy(omesh)
#     mesh.compute_vertex_normals()
#     line = o3d.geometry.LineSet.create_from_triangle_mesh(omesh)

#     # print(np.random.randint(5, size=1700))

#     # mesh.triangle.colors = o3d.core.Tensor(COLOR[np.random.randint(5, size=1700)])
    

#     o3d.visualization.draw([mesh])




import open3d as o3d

if __name__ == "__main__":
    o3d.visualization.webrtc_server.enable_webrtc()
    cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    o3d.visualization.draw(cube_red)