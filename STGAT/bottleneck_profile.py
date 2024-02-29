# import torch
# import torch.nn as nn
# from torch.profiler import profile, record_function, ProfilerActivity
# print(torch.get_num_threads())
# import threading
# import multiprocess
# multiprocess.set_start_method("spawn", force=True)
# torch.set_num_interop_threads(8)
# torch.set_num_threads(8)
# # Define a simple model for demonstration
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(64 * 56 * 56, 10)  # Adjust size for your input dimensions

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x

# # Set up the model and data
# model = MyModel()
# input = torch.randn(1, 3, 224, 224)  # Example input; adjust to your model's needs

# # Move model and input to the appropriate device (GPU if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# input = input.to(device)
# num_threads = threading.active_count()
# print(f"Number of active threads: {num_threads}")
# # Profile the model
# # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
# #     with record_function("model_inference"):
# #         model(input)

# # # Print out profiler results
# # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# # # Export profiler data for visualization
# # prof.export_chrome_trace("trace.json")
import threading
import time

n = 2

def crazy():
    global n
    while True:
        n = n*2
        print(n)
        time.sleep(1) # pause

threads = []
for i in range(4):
    t = threading.Thread(target=crazy)
    threads.append(t)
    t.start()