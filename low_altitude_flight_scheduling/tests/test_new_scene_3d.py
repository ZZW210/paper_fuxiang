from src.flight_plan import FlightPlan
from src.grid import AirspaceGrid
from src.visualization import write_fata_3d_html


def test_fata_3d_html_accepts_visual_emphasis_options(tmp_path):
    grid = AirspaceGrid(shape=(4, 4, 2), cell_size=(100, 100, 30), obstacle_ratio=0.0)
    path = [(0, 0, 1), (1, 1, 1), (2, 2, 1)]
    plans = [
        FlightPlan(fid, path[0], path[-1], path, 0.0, [0.0, 10.0, 20.0], [10.0, 10.0], 0.0, 20.0)
        for fid in (0, 1)
    ]
    output = tmp_path / "scene.html"
    write_fata_3d_html(
        grid,
        plans,
        [],
        output,
        "scene",
        highlight_ids=[plans[0].id],
        dim_other_routes=True,
        sidebar_html="<b>metrics</b>",
    )
    assert output.exists()
