#include "cp_fee.h"

#include <cstring>

static bool same(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int main() {
    const char* user = "prl1ptestwalletforfeescheduler";
    const char* project = "prl1p7arlanjq02vm70kfaq8vmgv85gu3fuu5wsagwng7n9t3zdmqfnlqulhkfx";
    cp_fee_init(user, 1);
    cp_fee_set_tiles_per_matrix(10);
    if (!cp_fee_enabled() || !same(cp_fee_wallet(), user)) return 1;
    cp_fee_on_authorized();
    cp_fee_note_tiles(1000);
    cp_fee_prepare_matrix();
    if (!cp_fee_needs_switch() || !same(cp_fee_wallet(), project)) return 2;
    cp_fee_on_authorized();
    cp_fee_note_tiles(10);
    if (!same(cp_fee_wallet(), user)) return 3;

    cp_fee_init(project, 1);
    if (cp_fee_enabled() || !same(cp_fee_wallet(), project)) return 4;
    return 0;
}
