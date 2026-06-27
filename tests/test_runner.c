#include "test_utils.h"

int main() {
    int passed = 0, failed = 0, total = 0;

    printf("\n"
           "+----------------------------------------------------------------+\n"
           "|                    TREN FRAMEWORK TESTS                        |\n"
           "+----------------------------------------------------------------+\n"
           "\n");

    printf("\n Tensor Basic Operations:\n"
           "------------------------------------------------------------------\n");
    test_tensor_basic_suite(&passed, &failed, &total);

    printf("\n Unary Operations:\n"
           "------------------------------------------------------------------\n");
    test_ops_unary_suite(&passed, &failed, &total);

    printf("\n Binary Operations:\n"
           "------------------------------------------------------------------\n");
    test_ops_binary_suite(&passed, &failed, &total);

    printf("\n Autograd:\n"
           "------------------------------------------------------------------\n");
    test_autograd_suite(&passed, &failed, &total);

    printf("\n Memory Management:\n"
           "------------------------------------------------------------------\n");
    test_memory_suite(&passed, &failed, &total);

    printf("\n"
           "+----------------------------------------------------------------+\n"
           "|                         TEST SUMMARY                           |\n"
           "+----------------------------------------------------------------+\n");
    printf("|  Tests passed: %d/%d                                           |\n", passed, total);
    printf("+----------------------------------------------------------------+\n\n");

    test_manual();
}