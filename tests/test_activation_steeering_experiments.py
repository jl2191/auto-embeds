# import pytest
# import torch
# import transformer_lens as tl
# from torch.utils.data import DataLoader, TensorDataset

# from auto_steer.steering_utils import (
#     calc_cos_sim_acc,
#     create_data_loaders,
#     evaluate_accuracy,
#     initialize_transform_and_optim,
#     run_and_gather_acts,
#     save_acts,
#     tokenize_texts,
#     train_transform,
# )


# # Mock data and model setup for tests
# @pytest.fixture
# def mock_model():
#     model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
#     return model

# @pytest.fixture
# def mock_data_loader():
#     en_embeds = torch.rand((10, 768))
#     fr_embeds = torch.rand((10, 768))
#     dataset = TensorDataset(en_embeds, fr_embeds)
#     loader = DataLoader(dataset, batch_size=2)
#     return loader

# def test_tokenize_texts(mock_model):
#     en_fr_pairs = [["hello", "bonjour"], ["world", "monde"]]
#     en_toks, en_attn_mask, fr_toks, fr_attn_mask = tokenize_texts(
#         mock_model,
#         en_fr_pairs,
#         padding_side="left",
#         single_tokens_only=True,
#         discard_if_same=True,
#         min_length=3,
#         capture_diff_case=True,
#         capture_space=True,
#         capture_no_space=True
#     )
#     assert en_toks.shape == fr_toks.shape
#     assert en_attn_mask.shape == fr_attn_mask.shape

# def test_create_data_loaders(mock_data_loader):
#     train_loader, test_loader = create_data_loaders(
#         torch.rand((10, 768)),
#         torch.rand((10, 768)),
#         batch_size=2,
#         train_ratio=0.8,
#     )
#     assert isinstance(train_loader, DataLoader)
#     assert isinstance(test_loader, DataLoader)

# def test_initialize_transform_and_optim(mock_model):
#     initial_rotation, optim = initialize_transform_and_optim(
#         d_model=768, transformation="linear_map", lr=0.0002, device=mock_model.cfg.device,
#     )
#     assert 'Linear' in initial_rotation.__class__.__name__
#     assert 'Adam' in optim.__class__.__name__

# def test_train_transform(mock_model, mock_data_loader):
#     initial_rotation, optim = initialize_transform_and_optim(
#         d_model=768, transformation="linear_map", lr=0.0002, device=mock_model.cfg.device,
#     )
#     learned_rotation, loss_history = train_transform(
#         mock_model, mock_data_loader, initial_rotation, optim, 1, device=mock_model.cfg.device,
#     )
#     assert len(loss_history) > 0

# def test_evaluate_accuracy(mock_model, mock_data_loader):
#     initial_rotation, _ = initialize_transform_and_optim(
#         d_model=768, transformation="linear_map", lr=0.0002, device=mock_model.cfg.device,
#     )
#     accuracy = evaluate_accuracy(
#         mock_model, mock_data_loader, initial_rotation, exact_match=False, device=mock_model.cfg.device,
#     )
#     assert isinstance(accuracy, float)

# def test_calc_cos_sim_acc(mock_model, mock_data_loader):
#     initial_rotation, _ = initialize_transform_and_optim(
#         d_model=768, transformation="linear_map", lr=0.0002, device=mock_model.cfg.device,
#     )
#     accuracy = calc_cos_sim_acc(mock_data_loader, initial_rotation, device=mock_model.cfg.device)
#     assert isinstance(accuracy, float)

# def test_run_and_gather_acts(mock_model, mock_data_loader):
#     en_acts, fr_acts = run_and_gather_acts(mock_model, mock_data_loader, layers=[0, 1])
#     assert isinstance(en_acts, dict)
#     assert isinstance(fr_acts, dict)

# def test_save_acts(tmp_path, mock_model, mock_data_loader):
#     en_acts, fr_acts = run_and_gather_acts(mock_model, mock_data_loader, layers=[0, 1])
#     save_acts(tmp_path, "test", en_acts, fr_acts)
#     assert (tmp_path / "test-en-layers-[0, 1].pt").exists()
#     assert (tmp_path / "test-fr-layers-[0, 1].pt").exists()
