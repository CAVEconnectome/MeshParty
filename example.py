import os


HOME = os.path.expanduser("~")
PACKAGE_PATH = "%s/../" % os.path.dirname(os.path.abspath(__file__))
EXAMPLE_PATH = "%s/example/" % PACKAGE_PATH

if not os.path.exists(EXAMPLE_PATH):
    os.makedirs(EXAMPLE_PATH)


def save_local_views(mesh, n_points, n_samples, save_name, pc_align=True):
    save_folder = "%s/%s/" % (EXAMPLE_PATH, save_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i_sample in range(n_samples):
        local_mesh = mesh.get_local_mesh(n_points=n_points,
                                         pc_align=pc_align)

        save_path = "%s/sample%d.obj" % (save_folder, i_sample)

        local_mesh.write_to_file(save_path)



if __name__ == "__main__":
    from meshparty import trimesh_io
    mm = trimesh_io.MeshMeta()
    mesh = mm.mesh("%s/MeshParty/example/3205058_m.obj" % HOME)

    save_local_views(mesh, n_points=2000, n_samples=10,
                     save_name="3205058_m_local_500")
