import time
import mujoco as mj

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.mujoco_renderer import MuJoCoViewer


class TurtlebotFactorySim:
    def __init__(self):
        # === tb3_factory.xml 절대/상대 경로 설정 ===
        script_path = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(script_path)
        PROJECT_ROOT = os.path.dirname(scripts_dir)    # /data/jinsup/js_mujoco

        factory_scene_path = os.path.join(
            PROJECT_ROOT,
            "asset",
            "robotis_tb3",
            "tb3_factory.xml",      
        )

        print(f"Loading scene from: {factory_scene_path}")

        # === SceneCreator 쓰지 말고, xml 파일 그대로 로드 ===
        model = mj.MjModel.from_xml_path(factory_scene_path)
        self.data = mj.MjData(model)

        # 기존 MuJoCoViewer 그대로 사용
        self.simulator = MuJoCoViewer(model, self.data)

    def start_rendering(self):
        try:
            while not self.simulator.should_close():
                time_prev = self.data.time
                while self.data.time - time_prev < 1.0 / 60.0:
                    self.simulator.step_simulation()

                # 왼쪽: 메인뷰 + IMU, 오른쪽: 로봇 카메라
                self.simulator.render_main(overlay_type="imu")  # "robot_pose" 도 가능
                self.simulator.render_robot()
                self.simulator.poll_events()
        except Exception as e:
            print(f"\n시뮬레이션을 종료합니다. {e}")
        finally:
            self.simulator.terminate()


if __name__ == "__main__":
    sim = TurtlebotFactorySim()
    sim.start_rendering()
