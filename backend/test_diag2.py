import sys; sys.path.insert(0, '.');
from main import load_model_diagnostics;
import traceback
print('testing inside backend')
try:
    load_model_diagnostics()
except:
    traceback.print_exc()
