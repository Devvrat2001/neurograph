/*
 * examples/capi/example.c — P1-08 sample usage of neurograph_rt.
 *
 * This file is intentionally pure C89 — no // comments, no mixed
 * decl/stmt, no bool. The build integrates it as a real target so a
 * C-incompatible construct in the header would break the build. That
 * covers the "header compiles as C" acceptance criterion without a
 * separate static-check CI job.
 *
 * Usage:
 *   example <bundle_dir>
 *
 * Reads one row from stdin as "f1=0.1,f2=0.2,f3=0.3", runs inference,
 * prints the result. Trivial on purpose — the goal is to demonstrate
 * the minimal call sequence, not to be a production tool. Customers
 * adapt this as a starting point.
 */

#include "neurograph/capi/neurograph_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char* where, ng_status st) {
    fprintf(stderr, "example: %s: status=%d last_error=%s\n",
            where, (int)st, ng_last_error());
    exit(1);
}

int main(int argc, char** argv) {
    ng_model* model = NULL;
    ng_status st;
    int       nf;
    int       nc;
    int       pred;
    const char* ver;
    /* Hard-coded demo row as JSON. Customers replace this with their
     * own row from wherever. */
    const char* row_json = "{\"f1\":1.0,\"f2\":1.0,\"f3\":-1.0}";

    if (argc < 2) {
        fprintf(stderr, "usage: %s <bundle_dir>\n", argv[0]);
        return 2;
    }

    st = ng_model_load(argv[1], &model);
    if (st != NG_OK) die("ng_model_load", st);

    nf  = ng_model_num_features(model);
    nc  = ng_model_num_classes(model);
    ver = ng_model_version(model);
    printf("loaded bundle: features=%d classes=%d version=%s\n",
           nf, nc, ver ? ver : "");

    if (nc > 0) {
        /* Classification path. */
        double* probs = (double*)malloc(sizeof(double) * (size_t)nc);
        if (!probs) {
            ng_model_free(model);
            fprintf(stderr, "example: OOM\n");
            return 1;
        }
        st = ng_predict_json(model, row_json, probs, &pred);
        if (st != NG_OK) {
            free(probs);
            ng_model_free(model);
            die("ng_predict_json", st);
        }
        {
            const char* lbl = ng_model_class_label(model, pred);
            int k;
            printf("predicted: id=%d label=%s\n", pred, lbl ? lbl : "");
            for (k = 0; k < nc; ++k) {
                printf("  p[%d]=%.6f\n", k, probs[k]);
            }
        }
        free(probs);
    } else {
        /* Regression path: probs[0] carries the scalar prediction. */
        double value = 0.0;
        st = ng_predict_json(model, row_json, &value, NULL);
        if (st != NG_OK) {
            ng_model_free(model);
            die("ng_predict_json", st);
        }
        printf("predicted value: %.6f\n", value);
    }

    ng_model_free(model);
    return 0;
}
