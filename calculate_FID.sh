base_dir=$1
cd $base_dir
mkdir -p "real_A" "real_B" "fake_A" "fake_B"
cp images/*real_A.png real_A
cp images/*real_B.png real_B
cp images/*fake_A.png fake_A
cp images/*fake_B.png fake_B
echo "FID(real A, translated B->A)"
python -m pytorch_fid "${base_dir}real_A" "${base_dir}fake_A" --device cuda:0
echo "FID(real B, translated A->B)"
python -m pytorch_fid "${base_dir}real_B" "${base_dir}fake_B" --device cuda:0
rm -rf "real_A"
rm -rf "fake_A"
rm -rf "real_B"
rm -rf "fake_B"
