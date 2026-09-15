"""Small regression check for chronological boundaries and ranking exclusion."""
import numpy as np
import pandas as pd
from train import rolling_masks, TIME, TARGET
from ensemble import blend_alpha

rows = pd.DataFrame({TIME: pd.to_datetime(['2025-06-30 23:59:59', '2025-07-01',
    '2025-07-31 23:59:59', '2025-08-01', '2025-06-01'], utc=True, format='mixed'),
    TARGET: [1., 2., 3., 4., 5.], 'source': ['training']*4 + ['ranking']})
train, valid = rolling_masks(rows, '2025-07')
np.testing.assert_array_equal(train, [True, False, False, False, False])
np.testing.assert_array_equal(valid, [False, True, True, False, False])
changed = rows.copy()
changed.loc[~train, TARGET] = 999999.
np.testing.assert_array_equal(rolling_masks(changed, '2025-07')[0], train)
try:
    rolling_masks(rows, '2025-12')
except ValueError:
    pass
else:
    raise AssertionError('Reserved December was accepted')
np.testing.assert_allclose(blend_alpha(np.array([0., 0.]), np.array([2., 4.]), np.array([1., 2.])), .5)
print('PASS: strict date boundary, future/ranking exclusion, December reservation, blend optimizer')
