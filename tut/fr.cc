import tensor_comprehensions as tc
import torch
lang = """
def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (0) {
  0(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
}

"""
N, C1, C2, C3, H, W = 32, 512, 8, 2, 28, 28
tensordot = tc.define(lang, name="tensordot")
I0, I1 = torch.randn(N, C1, C2, H, W).cuda(), torch.randn(N, C2, C3, H, W).cuda()
best_options = tensordot.autotune(I0, I1, cache=True)
out = tensordot(I0, I1, options=best_options)



//TEST
TEST(TensorDot, SimpleAutotune){
  std::string tc = R"TC(
def tensordot(float(N, C1, C2, H, W) I0,
              float(N, C2, C3, H, W) I1) -> (0)
{
  0(n, c1, c3, h3, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC";

  at::Tensor I0 = at::CUDA(at::kFloat).rand(32, 8, 16, 17, 25);
  at::Tensor I1 = at::CUDA(at::kFloat).rand(32, 16, 2, 17, 25);

  auto naiveOptions = Backend::MappingOptionsType::makeNaiveMappingOptions();
  tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>
    geneticAutotuneATen(tc);
  auto bestOption =
    geneticAutotuneATen.tune("tensordot", {I0, I1}, {naiveOptions});

  auto pExecutor =
    tc::aten::compile<Backend>(tc, "tensordot", {I0, I1}, bestOption[0]);
  auto outputs = tc::aten::prepareOutputs(tc, "tensordot", {I0, I1});
  auto timings = tc::aten::profile(*pExecutor, {I0, I1}, outputs);
  std::cuut << "tensordot size I0: " << I0.sizes() << ", "
            << "size I1: " << I1.sizes()
	    << " ran in: " << timings.kernelRuntime.toMicroSeconds() << "us\n";
}


for (auto sizes : std::vector<std::pair<at::IntList, at::IntList>>{
  {{4, 9, 7, 16, 14}, {4, 7, 3, 16, 14}},
  {{8, 5, 11, 10, 10},{9, 11, 16, 10, 10}},
}){
at::Tensor I0 = makeATenTensor<Backend>(sizes.first);
at::Tensor I1 = makeATenTensor<Backend>(sizes.second);
auto pExecutor =
  tc::aten::compile<Backend>(tc, "tensordot", {I0, I1}, bestOption[0]);
auto outputs = tc::aten::prepareOutputs(tc, "tensordot", {I0, I1});
auto timings = tc::aten::profile(*pExecutor, {I0, I1}, outputs);
std::cout << "tensordot size I0:" << I0.sizes() << ", "
	  << "size I1: " << I1.sizes()
	  << " ran in: " << timings.kernelRuntime.toMicroSeconds()
	  << "us\n";
}

// build$ ./examples/example_simple



