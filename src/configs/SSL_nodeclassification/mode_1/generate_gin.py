names = ["AutoEncoding","GAE","VGAE","ARGA","ARGVA","SuperGATSSL",
         "DenoisingLinkReconstruction", "EdgeMask","GRACE",
         "GCA","BGRL","SelfGNNSplit","SelfGNNPPR","SelfGNNLDP",
         "SelfGNNStandard","GBT","MERIT"]
for name in names:
  print(f'#{name}')
  print(f'@GCN_{name}_JL/NNNodeBenchmarkerSSL,')
  print(f'@GCN_{name}_PF/NNNodeBenchmarkerSSL,')
  print(f'@GCN_{name}_URL/NNNodeBenchmarkerSSL,')
  print()
  print(f'@GAT_{name}_JL/NNNodeBenchmarkerSSL,')
  print(f'@GAT_{name}_PF/NNNodeBenchmarkerSSL,')
  print(f'@GAT_{name}_URL/NNNodeBenchmarkerSSL,')
  print()
  print(f'@GIN_{name}_JL/NNNodeBenchmarkerSSL,')
  print(f'@GIN_{name}_PF/NNNodeBenchmarkerSSL,')
  print(f'@GIN_{name}_URL/NNNodeBenchmarkerSSL,')
  print()
  print(f'@APPNP_{name}_JL/NNNodeBenchmarkerSSL,')
  print(f'@APPNP_{name}_PF/NNNodeBenchmarkerSSL,')
  print(f'@APPNP_{name}_URL/NNNodeBenchmarkerSSL,')
  print()